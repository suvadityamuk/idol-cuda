#include "safetensors.h"

#include <cstdint>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace idol {
namespace {

// A tiny JSON reader, sufficient for safetensors headers (objects, arrays, strings, integers).
struct Json {
  enum Kind { Null, Num, Str, Arr, Obj } kind = Null;
  long long num = 0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;
  const Json& at(const std::string& k) const {
    for (auto& kv : obj)
      if (kv.first == k) return kv.second;
    throw std::runtime_error("json: missing key " + k);
  }
};

struct Parser {
  const char* p;
  const char* end;
  void ws() {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
  }
  std::string string() {
    if (*p != '"') throw std::runtime_error("json: expected string");
    ++p;
    std::string s;
    while (p < end && *p != '"') {
      if (*p == '\\') {
        ++p;
        switch (*p) {
          case 'n': s += '\n'; break;
          case 't': s += '\t'; break;
          case 'r': s += '\r'; break;
          case 'b': s += '\b'; break;
          case 'f': s += '\f'; break;
          case 'u': {  // keep ASCII only
            unsigned v = std::stoul(std::string(p + 1, p + 5), nullptr, 16);
            s += static_cast<char>(v < 128 ? v : '?');
            p += 4;
            break;
          }
          default: s += *p;
        }
        ++p;
      } else {
        s += *p++;
      }
    }
    ++p;
    return s;
  }
  Json value() {
    ws();
    Json j;
    if (*p == '{') {
      j.kind = Json::Obj;
      ++p;
      ws();
      if (*p == '}') { ++p; return j; }
      while (true) {
        ws();
        std::string k = string();
        ws();
        if (*p != ':') throw std::runtime_error("json: expected ':'");
        ++p;
        j.obj.emplace_back(k, value());
        ws();
        if (*p == ',') { ++p; continue; }
        if (*p == '}') { ++p; break; }
        throw std::runtime_error("json: bad object");
      }
    } else if (*p == '[') {
      j.kind = Json::Arr;
      ++p;
      ws();
      if (*p == ']') { ++p; return j; }
      while (true) {
        j.arr.push_back(value());
        ws();
        if (*p == ',') { ++p; continue; }
        if (*p == ']') { ++p; break; }
        throw std::runtime_error("json: bad array");
      }
    } else if (*p == '"') {
      j.kind = Json::Str;
      j.str = string();
    } else if (*p == 'n') {
      p += 4;
    } else {
      j.kind = Json::Num;
      char* e;
      j.num = std::strtoll(p, &e, 10);
      p = e;
      if (p < end && (*p == '.' || *p == 'e' || *p == 'E')) {  // skip fractional part (unused here)
        while (p < end && (std::isdigit(*p) || *p == '.' || *p == 'e' || *p == 'E' || *p == '-' || *p == '+')) ++p;
      }
    }
    return j;
  }
};

at::ScalarType parse_dtype(const std::string& s) {
  if (s == "F32") return at::kFloat;
  if (s == "F16") return at::kHalf;
  if (s == "BF16") return at::kBFloat16;
  if (s == "F64") return at::kDouble;
  if (s == "I64") return at::kLong;
  if (s == "I32") return at::kInt;
  if (s == "I16") return at::kShort;
  if (s == "I8") return at::kChar;
  if (s == "U8") return at::kByte;
  if (s == "BOOL") return at::kBool;
  if (s == "C64") return at::kComplexFloat;
  throw std::runtime_error("safetensors: unsupported dtype " + s);
}

const char* dtype_name(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "F32";
    case at::kHalf: return "F16";
    case at::kBFloat16: return "BF16";
    case at::kDouble: return "F64";
    case at::kLong: return "I64";
    case at::kInt: return "I32";
    case at::kShort: return "I16";
    case at::kChar: return "I8";
    case at::kByte: return "U8";
    case at::kBool: return "BOOL";
    case at::kComplexFloat: return "C64";
    default: throw std::runtime_error("safetensors: unsupported dtype");
  }
}

std::string escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o;
}

}  // namespace

std::map<std::string, at::Tensor> load_safetensors(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  uint64_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 8);
  std::string header(n, '\0');
  f.read(header.data(), n);
  Parser ps{header.data(), header.data() + header.size()};
  Json root = ps.value();
  f.seekg(0, std::ios::end);
  const uint64_t total = static_cast<uint64_t>(f.tellg());
  const uint64_t base = 8 + n;
  std::vector<char> data(total - base);
  f.seekg(base);
  f.read(data.data(), data.size());
  std::map<std::string, at::Tensor> out;
  for (auto& kv : root.obj) {
    if (kv.first == "__metadata__") continue;
    const Json& e = kv.second;
    at::ScalarType dt = parse_dtype(e.at("dtype").str);
    std::vector<int64_t> shape;
    for (auto& d : e.at("shape").arr) shape.push_back(d.num);
    const auto& off = e.at("data_offsets").arr;
    const int64_t begin = off[0].num, endo = off[1].num;
    at::Tensor t = at::empty(shape, at::TensorOptions().dtype(dt));
    if (endo - begin != static_cast<int64_t>(t.nbytes())) throw std::runtime_error("safetensors: size mismatch " + kv.first);
    if (t.nbytes()) std::memcpy(t.data_ptr(), data.data() + begin, t.nbytes());
    out.emplace(kv.first, t);
  }
  return out;
}

void save_safetensors(const std::string& path, const std::map<std::string, at::Tensor>& tensors,
                      const std::map<std::string, std::string>& metadata) {
  std::ostringstream h;
  h << "{";
  bool first = true;
  if (!metadata.empty()) {
    h << "\"__metadata__\":{";
    bool f2 = true;
    for (auto& kv : metadata) {
      if (!f2) h << ",";
      f2 = false;
      h << "\"" << escape(kv.first) << "\":\"" << escape(kv.second) << "\"";
    }
    h << "}";
    first = false;
  }
  std::vector<at::Tensor> cpu;
  uint64_t offset = 0;
  for (auto& kv : tensors) {
    at::Tensor t = kv.second.detach().to(at::kCPU).contiguous();
    cpu.push_back(t);
    if (!first) h << ",";
    first = false;
    h << "\"" << escape(kv.first) << "\":{\"dtype\":\"" << dtype_name(t.scalar_type()) << "\",\"shape\":[";
    for (int64_t i = 0; i < t.dim(); ++i) h << (i ? "," : "") << t.size(i);
    h << "],\"data_offsets\":[" << offset << "," << offset + t.nbytes() << "]}";
    offset += t.nbytes();
  }
  h << "}";
  std::string header = h.str();
  while (header.size() % 8) header += ' ';
  std::ofstream f(path, std::ios::binary);
  uint64_t n = header.size();
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(header.data(), header.size());
  for (auto& t : cpu)
    if (t.nbytes()) f.write(static_cast<const char*>(t.data_ptr()), t.nbytes());
  if (!f) throw std::runtime_error("failed writing " + path);
}

}  // namespace idol

#pragma once

/** 项目平铺标量配置读取，必填项缺失、重复、未知或数值未完整解析时立即报错。 */

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <filesystem>

namespace mm::detail {

// 【非核心】裁去标量两端空白，保留值内部的内容。
inline std::string trim_config_scalar(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

// 【非核心】两类C++入口共用读取规则，各自声明完整字段集合。
inline std::map<std::string, std::string> read_flat_config(
        const std::filesystem::path& filename, const std::set<std::string>& required) {
    std::ifstream input(filename);
    if (!input) {
        throw std::runtime_error("cannot open config: " + filename.string());
    }
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(input, line)) {
        line = trim_config_scalar(line.substr(0, line.find('#')));
        if (line.empty()) {
            continue;
        }
        const auto separator = line.find(':');
        if (separator == std::string::npos) {
            throw std::runtime_error("expected scalar key: value");
        }
        const auto key = trim_config_scalar(line.substr(0, separator));
        auto value = trim_config_scalar(line.substr(separator + 1));
        if (!required.contains(key)) {
            throw std::runtime_error("unknown config field: " + key);
        }
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        if (value.empty() || !fields.emplace(key, value).second) {
            throw std::runtime_error("empty or duplicate config field: " + key);
        }
    }
    for (const auto& key : required) {
        if (!fields.contains(key)) {
            throw std::runtime_error("missing required config field: " + key);
        }
    }
    return fields;
}

// 【非核心】数值必须完整消费整个标量，避免将100ms等错误配置默认为100。
template <typename Value>
Value config_number(const std::map<std::string, std::string>& fields, const std::string& key) {
    const auto& text = fields.at(key);
    if constexpr (std::is_unsigned_v<Value>) {
        if (text.starts_with('-')) {
            throw std::runtime_error("negative unsigned config field: " + key);
        }
    }
    std::istringstream input(text);
    Value value{};
    if (!(input >> value)) {
        throw std::runtime_error("invalid numeric config field: " + key);
    }
    input >> std::ws;
    if (!input.eof()) {
        throw std::runtime_error("trailing characters in numeric config field: " + key);
    }
    return value;
}

} // namespace mm::detail

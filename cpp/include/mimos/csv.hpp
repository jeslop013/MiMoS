// Minimal CSV readers for the MiMoS input formats (no quoting, no header).
//
// Every line is read with a trailing '\r' removed, so files written with CRLF (Python's
// csv.writer default) parse identically on Windows and macOS. Blank lines are skipped.
#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mimos::csv {

inline std::ifstream open(const std::string& file) {
    std::ifstream in(file);
    if (!in.is_open()) throw std::runtime_error("MiMoS: cannot open input file " + file);
    return in;
}

// Reads the next line without its line ending. Returns false at end of file.
inline bool next_line(std::istream& in, std::string& line) {
    if (!std::getline(in, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

// Splits a line on commas and converts every field with `parse`.
template <typename T, typename Parse>
std::vector<T> parse_row(const std::string& line, Parse parse) {
    std::vector<T> row;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) row.push_back(parse(field));
    return row;
}

// One vector per non-blank line.
template <typename T, typename Parse>
std::vector<std::vector<T>> read_rows(const std::string& file, Parse parse) {
    auto in = open(file);
    std::vector<std::vector<T>> rows;
    std::string line;
    while (next_line(in, line))
        if (!line.empty()) rows.push_back(parse_row<T>(line, parse));
    return rows;
}

// Lines split into exactly `n_fields` strings: the first n_fields-1 at commas, the last one
// is the remainder of the line (getline(',') ... getline('\n')).
inline std::vector<std::vector<std::string>> read_fields(const std::string& file, int n_fields) {
    auto in = open(file);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (next_line(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> f;
        size_t start = 0;
        for (int k = 0; k < n_fields - 1; ++k) {
            size_t comma = line.find(',', start);
            if (comma == std::string::npos) throw std::runtime_error("MiMoS: too few fields in " + file);
            f.push_back(line.substr(start, comma - start));
            start = comma + 1;
        }
        f.push_back(line.substr(start));
        rows.push_back(f);
    }
    return rows;
}

}  // namespace mimos::csv

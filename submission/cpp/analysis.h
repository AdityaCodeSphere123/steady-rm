#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>

extern const std::unordered_set<std::string> STOPWORDS;
extern const std::unordered_set<std::string> INDEX_STOPWORDS;

const std::string& apply_stem(const std::string& w, uint8_t stemmer);

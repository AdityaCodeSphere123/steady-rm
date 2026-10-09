#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>

extern const std::unordered_set<std::string> STOPWORDS;
extern const std::unordered_set<std::string> INDEX_STOPWORDS;

// stemmer 0 keeps the word; any other value applies the Porter stemmer.
const std::string& apply_stem(const std::string& w, uint8_t stemmer);

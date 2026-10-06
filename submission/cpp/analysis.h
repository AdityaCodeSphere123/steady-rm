// Stopword lists and the Porter stemmer.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>

// Fox-style list: terms never used as expansion terms.
extern const std::unordered_set<std::string> STOPWORDS;
// Lucene English list: removed from queries.
extern const std::unordered_set<std::string> INDEX_STOPWORDS;

// stemmer 0 returns w unchanged; any other value applies Porter.
const std::string& apply_stem(const std::string& w, uint8_t stemmer);

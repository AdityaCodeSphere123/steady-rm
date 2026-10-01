// Query/index analysis: lowercase, split on non-alnum, fold dotted
// initials (U.S. → us), drop leftover length-1, optional stops, stem.
#pragma once

#include <string>
#include <unordered_set>

extern const std::unordered_set<std::string> STOPWORDS;
extern const std::unordered_set<std::string> INDEX_STOPWORDS;
// Articles/prepositions only. No WH-words, negations, or modals.
extern const std::unordered_set<std::string> MINIMAL_STOPWORDS;
extern const std::unordered_set<std::string> LUCENE_MINUS_WH;

const std::string& apply_stem(const std::string& w, uint8_t stemmer);

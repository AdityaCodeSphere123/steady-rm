#include "analysis.h"

#include <unordered_map>

const std::unordered_set<std::string> STOPWORDS = {
    "a", "about", "above", "across", "after", "again", "against", "all", "almost", "alone", "along",
    "already", "also", "although", "always", "among", "an", "and", "another", "any", "anybody", "anyone",
    "anything", "anywhere", "are", "area", "areas", "around", "as", "ask", "asked", "asking", "asks", "at",
    "away", "b", "back", "backed", "backing", "backs", "be", "became", "because", "become", "becomes",
    "been", "before", "began", "behind", "being", "beings", "best", "better", "between", "big", "both",
    "but", "by", "c", "came", "can", "cannot", "case", "cases", "certain", "certainly", "clear", "clearly",
    "come", "could", "d", "did", "differ", "different", "differently", "do", "does", "done", "down",
    "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended", "ending",
    "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone", "everything",
    "everywhere", "f", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four",
    "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave", "general",
    "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great",
    "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had", "has", "have", "having",
    "he", "her", "here", "herself", "high", "higher", "highest", "him", "himself", "his", "how", "however",
    "i", "if", "important", "in", "interest", "interested", "interesting", "interests", "into", "is",
    "it", "its", "itself", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows",
    "l", "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "like", "likely",
    "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member",
    "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "my", "myself",
    "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "newer", "newest", "next",
    "no", "nobody", "non", "noone", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of",
    "off", "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened", "opening",
    "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "our", "out", "over",
    "p", "part", "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed",
    "pointing", "points", "possible", "present", "presented", "presenting", "presents", "problem",
    "problems", "put", "puts", "q", "quite", "r", "rather", "really", "right", "room", "rooms", "s",
    "said", "same", "saw", "say", "says", "second", "seconds", "see", "seem", "seemed", "seeming",
    "seems", "sees", "several", "shall", "she", "should", "show", "showed", "showing", "shows", "side",
    "sides", "since", "small", "smaller", "smallest", "so", "some", "somebody", "someone", "something",
    "somewhere", "state", "states", "still", "such", "sure", "t", "take", "taken", "than", "that", "the",
    "their", "them", "then", "there", "therefore", "these", "they", "thing", "things", "think", "thinks",
    "this", "those", "though", "thought", "thoughts", "three", "through", "thus", "to", "today",
    "together", "too", "took", "toward", "turn", "turned", "turning", "turns", "two", "u", "under",
    "until", "up", "upon", "us", "use", "used", "uses", "v", "very", "w", "want", "wanted", "wanting",
    "wants", "was", "way", "ways", "we", "well", "wells", "went", "were", "what", "when", "where",
    "whether", "which", "while", "who", "whole", "whose", "why", "will", "with", "within", "without",
    "work", "worked", "working", "works", "would", "x", "y", "year", "years", "yet", "you", "your", "yours"
};

const std::unordered_set<std::string> INDEX_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers","herself",
    "it","its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were","be",
    "been","being","have","has","had","having","do","does","did","doing","a","an",
    "the","and","but","if","or","because","as","until","while","of","at","by","for",
    "with","about","against","between","into","through","during","before","after",
    "above","below","to","from","up","down","in","out","on","off","over","under",
    "again","further","then","once","here","there","when","where","why","how","all",
    "any","both","each","few","more","most","other","some","such","no","nor","not",
    "only","own","same","so","than","too","very","can","will","just","should","now"
};


class PorterStemmer {
    std::string b;
    int k = 0;   // index of the last character of the word
    int j = 0;   // general offset used by the step helpers

    bool cons(int i) const {
        switch (b[i]) {
            case 'a': case 'e': case 'i': case 'o': case 'u': return false;
            case 'y': return (i == 0) ? true : !cons(i - 1);
            default: return true;
        }
    }

    // m() measures the number of vowel-consonant sequences in b[0..j]
    int m() const {
        int n = 0, i = 0;
        for (;;) {
            if (i > j) return n;
            if (!cons(i)) break;
            i++;
        }
        i++;
        for (;;) {
            for (;;) {
                if (i > j) return n;
                if (cons(i)) break;
                i++;
            }
            i++;
            n++;
            for (;;) {
                if (i > j) return n;
                if (!cons(i)) break;
                i++;
            }
            i++;
        }
    }

    bool vowelinstem() const {
        for (int i = 0; i <= j; i++) if (!cons(i)) return true;
        return false;
    }

    bool doublec(int i) const {
        if (i < 1) return false;
        if (b[i] != b[i - 1]) return false;
        return cons(i);
    }

    // cvc(i): consonant-vowel-consonant where the final consonant is not w, x or y
    bool cvc(int i) const {
        if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
        char ch = b[i];
        if (ch == 'w' || ch == 'x' || ch == 'y') return false;
        return true;
    }

    bool ends(const char* s) {
        int len = static_cast<int>(std::strlen(s));
        if (len > k + 1) return false;
        if (b.compare(k - len + 1, len, s) != 0) return false;
        j = k - len;
        return true;
    }

    void setto(const char* s) {
        int len = static_cast<int>(std::strlen(s));
        b.replace(j + 1, k - j, s);
        k = j + len;
    }

    void r(const char* s) { if (m() > 0) setto(s); }

    void step1ab() {
        if (b[k] == 's') {
            if (ends("sses")) k -= 2;
            else if (ends("ies")) setto("i");
            else if (b[k - 1] != 's') k--;
        }
        if (ends("eed")) { if (m() > 0) k--; }
        else if ((ends("ed") || ends("ing")) && vowelinstem()) {
            k = j;
            if (ends("at")) setto("ate");
            else if (ends("bl")) setto("ble");
            else if (ends("iz")) setto("ize");
            else if (doublec(k)) {
                k--;
                char ch = b[k];
                if (ch == 'l' || ch == 's' || ch == 'z') k++;
            } else if (m() == 1 && cvc(k)) {
                setto("e");
            }
        }
    }

    void step1c() {
        if (ends("y") && vowelinstem()) b[k] = 'i';
    }

    void step2() {
        if (k == 0) return;
        switch (b[k - 1]) {
            case 'a': if (ends("ational")) { r("ate"); break; }
                      if (ends("tional"))  { r("tion"); break; }
                      break;
            case 'c': if (ends("enci"))    { r("ence"); break; }
                      if (ends("anci"))    { r("ance"); break; }
                      break;
            case 'e': if (ends("izer"))    { r("ize"); break; }
                      break;
            case 'l': if (ends("abli"))    { r("able"); break; }
                      if (ends("alli"))    { r("al"); break; }
                      if (ends("entli"))   { r("ent"); break; }
                      if (ends("eli"))     { r("e"); break; }
                      if (ends("ousli"))   { r("ous"); break; }
                      break;
            case 'o': if (ends("ization")) { r("ize"); break; }
                      if (ends("ation"))   { r("ate"); break; }
                      if (ends("ator"))    { r("ate"); break; }
                      break;
            case 's': if (ends("alism"))   { r("al"); break; }
                      if (ends("iveness")) { r("ive"); break; }
                      if (ends("fulness")) { r("ful"); break; }
                      if (ends("ousness")) { r("ous"); break; }
                      break;
            case 't': if (ends("aliti"))   { r("al"); break; }
                      if (ends("iviti"))   { r("ive"); break; }
                      if (ends("biliti"))  { r("ble"); break; }
                      break;
        }
    }

    void step3() {
        switch (b[k]) {
            case 'e': if (ends("icate")) { r("ic"); break; }
                      if (ends("ative")) { r(""); break; }
                      if (ends("alize")) { r("al"); break; }
                      break;
            case 'i': if (ends("iciti"))  { r("ic"); break; }
                      break;
            case 'l': if (ends("ical"))   { r("ic"); break; }
                      if (ends("ful"))    { r(""); break; }
                      break;
            case 's': if (ends("ness"))   { r(""); break; }
                      break;
        }
    }

    void step4() {
        if (k == 0) return;
        switch (b[k - 1]) {
            case 'a':
                if (ends("al")) break;
                return;
            case 'c':
                if (ends("ance") || ends("ence")) break;
                return;
            case 'e':
                if (ends("er")) break;
                return;
            case 'i':
                if (ends("ic")) break;
                return;
            case 'l':
                if (ends("able") || ends("ible")) break;
                return;
            case 'n':
                if (ends("ant") || ends("ement") || ends("ment") || ends("ent")) break;
                return;
            case 'o':
                if (ends("ion") && j >= 0 && (b[j] == 's' || b[j] == 't')) break;
                if (ends("ou")) break;
                return;
            case 's':
                if (ends("ism")) break;
                return;
            case 't':
                if (ends("ate") || ends("iti")) break;
                return;
            case 'u':
                if (ends("ous")) break;
                return;
            case 'v':
                if (ends("ive")) break;
                return;
            case 'z':
                if (ends("ize")) break;
                return;
            default:
                return;
        }
        if (m() > 1) k = j;
    }

    void step5() {
        j = k;
        if (b[k] == 'e') {
            int a = m();
            if (a > 1 || (a == 1 && !cvc(k - 1))) k--;
        }
        if (b[k] == 'l' && doublec(k) && m() > 1) k--;
    }

public:
    std::string stem(const std::string& word) {
        if (word.size() <= 2) return word;
        b = word;
        k = static_cast<int>(word.size()) - 1;
        step1ab();
        if (k > 0) step1c();
        if (k > 0) step2();
        if (k > 0) step3();
        if (k > 0) step4();
        if (k > 0) step5();
        return b.substr(0, k + 1);
    }
};

thread_local PorterStemmer g_porter;
// One cache per mode: the stemmer can change between prepare() calls.
thread_local std::unordered_map<std::string, std::string> g_stem_cache[2];

const std::string& apply_stem(const std::string& w, uint8_t stemmer) {
    auto& cache = g_stem_cache[stemmer ? 1 : 0];
    auto it = cache.find(w);
    if (it != cache.end()) return it->second;
    return cache.emplace(w, stemmer ? g_porter.stem(w) : w).first->second;
}

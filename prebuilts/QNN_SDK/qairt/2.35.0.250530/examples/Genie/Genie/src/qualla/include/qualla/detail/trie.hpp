//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_TRIE_HPP
#define QUALLA_DETAIL_TRIE_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fmt/format.h"

namespace qualla {

class SequenceMatchTrie {
 private:
  // datastructures for trie and end-states
  struct TrieNode {
    std::unordered_map<char, std::unique_ptr<TrieNode>> data;
    TrieNode() {}

    std::string str() const {
      std::string s = "{ ";
      for (const auto& [c, n] : data) s += fmt::format("\"{:c}\": {:s}, ", c, n->str());
      if (s.size() > 2) s.erase(s.size() - 2);
      return s + "} ";
    }
  } _root;

  std::unordered_set<TrieNode*> _end_states;
  std::vector<TrieNode*> _cur_match_state;

 public:
  enum class MatchType { NO_MATCH, PARTIAL_MATCH, COMPLETE_MATCH };
  SequenceMatchTrie() { clear(); }
  SequenceMatchTrie(const std::vector<std::string>& sequences) {
    clear();
    build_trie(sequences);
  }

  void print_trie(TrieNode* node) { fprintf(stderr, "%s\n", node->str().c_str()); }

  void build_trie(const std::vector<std::string>& sequences) {
    // Construct the sequence trie for matching
    for (const std::string& sequence : sequences) {
      TrieNode* cur_node = &_root;
      for (const char c : sequence) {
        // Add character to the trie if it doesn't exist
        if (!cur_node->data.contains(c)) cur_node->data[c] = std::make_unique<TrieNode>();

        // Traverse to next node
        cur_node = cur_node->data[c].get();
      }

      // Add end state of iteration as goal state
      _end_states.insert(cur_node);
    }
  }

  MatchType process_next_char(const char c) {
    std::vector<TrieNode*> _next_match_state = {&_root};
    for (TrieNode* state : _cur_match_state) {
      if (!state->data.contains(c)) continue;

      TrieNode* next_state = state->data[c].get();
      if (_end_states.contains(next_state)) return MatchType::COMPLETE_MATCH;
      _next_match_state.push_back(next_state);
    }

    _cur_match_state = _next_match_state;
    if (_cur_match_state.size() > 1) return MatchType::PARTIAL_MATCH;
    return MatchType::NO_MATCH;
  }

  std::pair<MatchType, uint32_t> process_next_string(const std::string& s) {
    uint32_t matchStartIndex =
        s.size();  // index of generated token string from which stop sequence started
    uint32_t index = 0;
    for (const char c : s) {
      MatchType nextCharStatus = process_next_char(c);
      if (nextCharStatus != MatchType::NO_MATCH && matchStartIndex >= index)
        matchStartIndex = index;
      if (nextCharStatus == MatchType::COMPLETE_MATCH)
        return std::make_pair(MatchType::COMPLETE_MATCH, matchStartIndex);
      index++;
    }
    if (_cur_match_state.size() > 1)
      return std::make_pair(MatchType::PARTIAL_MATCH, matchStartIndex);
    return std::make_pair(MatchType::NO_MATCH, matchStartIndex);
  }

  bool empty() { return _root.data.empty(); }
  void reset() { _cur_match_state = {&_root}; }

  void clear() {
    _root = TrieNode();
    _end_states.clear();  // Clear out end states
    reset();
  }
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_TRIE_HPP

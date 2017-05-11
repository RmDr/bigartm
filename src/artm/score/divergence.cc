#include <cmath>
#include <map>
#include <algorithm>
#include <sstream>

#include "artm/core/exceptions.h"
#include "artm/core/protobuf_helpers.h"

#include "artm/score/divergence.h"

namespace artm {
namespace score {

Divergence::Divergence(const ScoreConfig& config) : ScoreCalculatorInterface(config) {
  config_ = ParseConfig<DivergenceScoreConfig>();
  std::stringstream ss;
  ss << ": model_type=" << config_.model_type();
  if (config_.has_dictionary_name())
    ss << ", dictionary_name=" << config_.dictionary_name();
  LOG(INFO) << "Divergence score calculator created" << ss.str();
}

void Divergence::AppendScore(
    const Item& item,
    const std::vector<artm::core::Token>& token_dict,
    const artm::core::PhiMatrix& p_wt,
    const artm::ProcessBatchesArgs& args,
    const std::vector<float>& theta,
    Score* score) {
  int topic_size = p_wt.topic_size();

  // the following code counts KL-divergence
  bool use_classes_from_model = false;
  if (config_.class_id_size() == 0) use_classes_from_model = true;

  std::map< ::artm::core::ClassId, float> class_weights;
  if (use_classes_from_model) {
    for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i)
      class_weights.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
  } else {
    for (auto& class_id : config_.class_id()) {
      for (int i = 0; (i < args.class_id_size()) && (i < args.class_weight_size()); ++i)
        if (class_id == args.class_id(i)) {
          class_weights.insert(std::make_pair(args.class_id(i), args.class_weight(i)));
          break;
        }
    }
  }
  bool use_class_id = !class_weights.empty();
	
  float n_d = 0;
  for (int token_index = 0; token_index < item.token_weight_size(); ++token_index) {
    float class_weight = 1.0f;
    if (use_class_id) {
      ::artm::core::ClassId class_id = token_dict[item.token_id(token_index)].class_id;
      auto iter = class_weights.find(class_id);
      if (iter == class_weights.end())
        continue;
      class_weight = iter->second;
    }

    n_d += class_weight * item.token_weight(token_index);
  }

  ::google::protobuf::int64 zero_words = 0;
  double raw = 0;

  std::shared_ptr<core::Dictionary> dictionary_ptr = nullptr;
  if (config_.has_dictionary_name())
    dictionary_ptr = dictionary(config_.dictionary_name());
  bool has_dictionary = dictionary_ptr != nullptr;

  bool use_document_unigram_model = true;
  if (config_.has_model_type()) {
    if (config_.model_type() == DivergenceScoreConfig_Type_UnigramCollectionModel) {
      if (has_dictionary) {
        use_document_unigram_model = false;
      } else {
        LOG(ERROR) << "Divergence was configured to use UnigramCollectionModel with dictionary " <<
           config_.dictionary_name() << ". This dictionary can't be found.";
        return;
      }
    }
  }

  std::vector<float> helper_vector(topic_size, 0.0f);
  for (int token_index = 0; token_index < item.token_weight_size(); ++token_index) {
    double sum = 0.0;
    const artm::core::Token& token = token_dict[item.token_id(token_index)];

    float class_weight = 1.0f;
    if (use_class_id) {
      auto iter = class_weights.find(token.class_id);
      if (iter == class_weights.end())
        continue;
      class_weight = iter->second;
    }

    float token_weight = class_weight * item.token_weight(token_index);
    if (token_weight == 0.0f) continue;

    int p_wt_token_index = p_wt.token_index(token);
    if (p_wt_token_index != ::artm::core::PhiMatrix::kUndefIndex) {
      p_wt.get(p_wt_token_index, &helper_vector);
      for (int topic_index = 0; topic_index < topic_size; topic_index++) {
        sum += theta[topic_index] * helper_vector[topic_index];
      }
    }
    if (sum == 0.0) {
      if (use_document_unigram_model) {
        sum = token_weight / n_d;
      } else {
        auto entry_ptr = dictionary_ptr->entry(token);
        bool failed = true;
        if (entry_ptr != nullptr && entry_ptr->token_value()) {
          sum = entry_ptr->token_value();
          failed = false;
        }
        if (failed) {
          LOG_FIRST_N(WARNING, 1)
                    << "Error in divergence dictionary for token " << token.keyword << ", class " << token.class_id
                    << " (and potentially for other tokens)"
                    << ". Verify that the token exists in the dictionary and it's value > 0. "
                    << "Document unigram model will be used for this token "
                    << "(and for all other tokens under the same conditions).";
          sum = token_weight / n_d;
        }
      }
      zero_words++;
    }

    raw        += token_weight * log(token_weight) - token_weight * log(sum);
  }

  // prepare results
  DivergenceScore divergence_score;
  divergence_score.set_raw(raw);
  divergence_score.set_zero_words(zero_words);
  AppendScore(divergence_score, score);
}

std::shared_ptr<Score> Divergence::CreateScore() {
  VLOG(1) << "Divergence::CreateScore()";
  return std::make_shared<DivergenceScore>();
}

void Divergence::AppendScore(const Score& score, Score* target) {
  std::string error_message = "Unable downcast Score to DivergenceScore";
  const DivergenceScore* perplexity_score = dynamic_cast<const DivergenceScore*>(&score);
  if (divergence_score == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  DivergenceScore* divergence_target = dynamic_cast<DivergenceScore*>(target);
  if (divergence_target == nullptr) {
    BOOST_THROW_EXCEPTION(::artm::core::InternalError(error_message));
  }

  divergence_target->set_raw(divergence_target->raw() +
                             divergence_score->raw());
  divergence_target->set_zero_words(divergence_target->zero_words() +
                                    divergence_score->zero_words());
  divergence_target->set_value(divergence_target->raw());

  VLOG(1) << ", raw=" << divergence_target->raw()
          << ", zero_words=" << divergence_target->zero_words();
}

}  // namespace score
}  // namespace artm

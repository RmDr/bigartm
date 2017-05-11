#ifndef SRC_ARTM_SCORE_DIVERGENCE_H_
#define SRC_ARTM_SCORE_DIVERGENCE_H_

#include <string>
#include <vector>

#include "artm/score_calculator_interface.h"

namespace artm {
namespace score {

class Divergence : public ScoreCalculatorInterface {
 public:
  explicit Divergence(const ScoreConfig& config);

  virtual bool is_cumulative() const { return true; }

  virtual std::shared_ptr<Score> CreateScore();

  virtual void AppendScore(const Score& score, Score* target);

  virtual void AppendScore(
      const Item& item,
      const std::vector<artm::core::Token>& token_dict,
      const artm::core::PhiMatrix& p_wt,
      const artm::ProcessBatchesArgs& args,
      const std::vector<float>& theta,
      Score* score);

  virtual ScoreType score_type() const { return ::artm::ScoreType_Divergence; }

 private:
  DivergenceScoreConfig config_;
};

}  // namespace score
}  // namespace artm

#endif  // SRC_ARTM_SCORE_DIVERGENCE_H_

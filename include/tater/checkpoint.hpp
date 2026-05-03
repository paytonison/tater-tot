#pragma once

#include "tater/data.hpp"
#include "tater/model.hpp"

#include <string>

namespace tater {

struct LoadedCheckpoint {
    TinyCharModel model;
    Vocabulary vocab;
};

void save_checkpoint(const std::string& path, const TinyCharModel& model, const Vocabulary& vocab);
LoadedCheckpoint load_checkpoint(const std::string& path);

} // namespace tater


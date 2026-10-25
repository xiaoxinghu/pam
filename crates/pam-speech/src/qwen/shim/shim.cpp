// A C interface to qwen3-asr.cpp's ForcedAligner, for `pam_speech::qwen`.
#include "forced_aligner.h"

#include <exception>
#include <string>
#include <vector>

extern "C" {

struct pam_fa_unit {
    const char * text;  // owned by the aligner until the next call
    float start;        // seconds from the clip's start
    float end;
};

struct pam_fa {
    qwen3_asr::ForcedAligner fa;
    qwen3_asr::alignment_result last;
    std::vector<pam_fa_unit> units;
    std::string error;
};

pam_fa * pam_fa_open(const char * model, int n_threads) {
    auto * a = new pam_fa();
    if (!a->fa.load_model(model)) {
        delete a;
        return nullptr;
    }
    a->fa.set_n_threads(n_threads);
    return a;
}

// Times `units`, separated by single spaces, against 16 kHz mono audio.
// A language other than Chinese, Japanese or Korean makes the runtime split
// the text at whitespace only, so it times exactly the units given; the model
// itself is never told a language. Returns the number of units timed, or -1
// with the reason in `pam_fa_error`.
int pam_fa_align(pam_fa * a, const float * samples, int n, const char * units, const pam_fa_unit ** out) {
    try {
        a->last = a->fa.align(samples, n, units, "units");
    } catch (const std::exception & e) {
        a->error = e.what();
        return -1;
    }
    if (!a->last.success) {
        a->error = a->last.error_msg;
        return -1;
    }
    a->units.clear();
    for (const auto & w : a->last.words) a->units.push_back({w.word.c_str(), w.start, w.end});
    *out = a->units.data();
    return (int)a->units.size();
}

const char * pam_fa_error(const pam_fa * a) { return a->error.c_str(); }

void pam_fa_close(pam_fa * a) { delete a; }

}

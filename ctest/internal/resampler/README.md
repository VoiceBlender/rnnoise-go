# Vendored Speex resampler

Copied verbatim from `VoiceBlender/internal/resampler` (which is itself a Go port
of the Speex arbitrary resampler by Jean-Marc Valin — the same author as RNNoise).
Licence and copyright notices are preserved in the source files.

It lives here because it sits behind an `internal/` path in the VoiceBlender
module and so cannot be imported across modules, and because the native-rate
quality harness needs *the* resampler VoiceBlender actually runs in production:
the reference chain it defines is "what you would get by resampling to 48 kHz and
back", so using a different resampler would measure the wrong thing.

Test-only. The `rnnoise-go` library itself does no resampling and does not depend
on this.

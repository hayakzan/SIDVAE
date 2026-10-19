#include "SidVaeDecoder.h"
#include "TableUtils.h"

#include <algorithm>
#include <cctype>
#include <sstream>

#if SIDVAE_WITH_ONNX
 #include <onnxruntime_cxx_api.h>
#endif

namespace sidvae
{
#if SIDVAE_WITH_ONNX
namespace
{
    Ort::Env& ortEnv()
    {
        static Ort::Env env (ORT_LOGGING_LEVEL_WARNING, "SIDVae");
        return env;
    }

    struct InputInfo
    {
        std::string name;
        std::vector<int64_t> shape;   // as declared; dynamic dims are <= 0
        int64_t perItem = -1;         // elements per batch item, -1 if unknown
    };

    int64_t elementsPerItem (const std::vector<int64_t>& s)
    {
        if (s.empty())      return -1;
        if (s.size() == 1)  return s[0] > 0 ? s[0] : -1;   // no batch dim
        int64_t p = 1;
        for (std::size_t i = 1; i < s.size(); ++i)
        {
            if (s[i] <= 0) return -1;
            p *= s[i];
        }
        return p;
    }

    // Turn a declared shape (possibly with dynamic dims) into a concrete one holding n elements.
    std::vector<int64_t> concreteShape (const std::vector<int64_t>& declared, int64_t n)
    {
        if (declared.size() == 1) return { n };
        if (declared.size() < 2)  return { 1, n };

        std::vector<int64_t> s = declared;
        s[0] = 1;                               // batch of one
        int unknownAt = -1;
        int64_t known = 1;
        for (std::size_t i = 1; i < s.size(); ++i)
        {
            if (s[i] <= 0)
            {
                if (unknownAt >= 0) return { 1, n };
                unknownAt = (int) i;
            }
            else known *= s[i];
        }
        if (unknownAt >= 0)
        {
            if (known == 0 || n % known != 0) return { 1, n };
            s[(std::size_t) unknownAt] = n / known;
        }
        else if (known != n) return { 1, n };
        return s;
    }

    std::string shapeToString (const std::vector<int64_t>& s)
    {
        std::ostringstream os;
        os << '[';
        for (std::size_t i = 0; i < s.size(); ++i) os << (i ? "," : "") << s[i];
        os << ']';
        return os.str();
    }

    bool looksLikeCondition (std::string n)
    {
        std::transform (n.begin(), n.end(), n.begin(), [] (unsigned char ch) { return (char) std::tolower (ch); });
        return n.find ("cond") != std::string::npos || n.find ("label") != std::string::npos || n == "c" || n == "y";
    }
}

struct SidVaeDecoder::Impl
{
    enum class Layout { TwoInputs, Concatenated };

    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);

    Layout layout = Layout::TwoInputs;
    std::string zName, cName, catName, outName;
    std::vector<int64_t> zShape, cShape, catShape;
    std::vector<float> zBuf, cBuf, catBuf;
};
#else
struct SidVaeDecoder::Impl {};
#endif

SidVaeDecoder::SidVaeDecoder()  = default;
SidVaeDecoder::~SidVaeDecoder() = default;

bool SidVaeDecoder::load (const std::filesystem::path& modelFile, const Config& c, std::string& error)
{
    ready = false;
    impl.reset();
    summary.clear();
    cfg = c;

#if SIDVAE_WITH_ONNX
    try
    {
        std::error_code ec;
        if (! std::filesystem::is_regular_file (modelFile, ec))
        {
            error = "model file not found: " + modelFile.string();
            return false;
        }

        auto im = std::make_unique<Impl>();

        Ort::SessionOptions so;
        so.SetIntraOpNumThreads (1);
        so.SetInterOpNumThreads (1);
        so.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);

        // path::c_str() is wchar_t* on Windows and char* elsewhere, matching ORTCHAR_T.
        im->session = std::make_unique<Ort::Session> (ortEnv(), modelFile.c_str(), so);

        Ort::AllocatorWithDefaultOptions alloc;
        std::vector<InputInfo> ins;

        for (std::size_t i = 0; i < im->session->GetInputCount(); ++i)
        {
            InputInfo info;
            info.name = im->session->GetInputNameAllocated (i, alloc).get();
            // Keep TypeInfo alive: the shape info is a view into it.
            const Ort::TypeInfo typeInfo = im->session->GetInputTypeInfo (i);
            const auto t = typeInfo.GetTensorTypeAndShapeInfo();
            if (t.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            {
                error = "input '" + info.name + "' is not float32";
                return false;
            }
            info.shape   = t.GetShape();
            info.perItem = elementsPerItem (info.shape);
            ins.push_back (std::move (info));
        }

        if (im->session->GetOutputCount() < 1)
        {
            error = "model has no outputs";
            return false;
        }
        im->outName = im->session->GetOutputNameAllocated (0, alloc).get();

        const int64_t L = cfg.latentDim, C = cfg.condDim;
        std::ostringstream io;

        if (ins.size() == 2)
        {
            int zi = -1;
            const bool sizesKnown = ins[0].perItem > 0 && ins[1].perItem > 0;

            if (sizesKnown && L != C)
            {
                if      (ins[0].perItem == L && ins[1].perItem == C) zi = 0;
                else if (ins[0].perItem == C && ins[1].perItem == L) zi = 1;
            }
            if (zi < 0)
            {
                const bool c0 = looksLikeCondition (ins[0].name), c1 = looksLikeCondition (ins[1].name);
                zi = (c0 && ! c1) ? 1 : 0;
            }
            const auto& zIn = ins[(std::size_t) zi];
            const auto& cIn = ins[(std::size_t) (1 - zi)];

            if ((zIn.perItem > 0 && zIn.perItem != L) || (cIn.perItem > 0 && cIn.perItem != C))
            {
                error = "input sizes don't match meta: '" + zIn.name + "' " + shapeToString (zIn.shape)
                      + ", '" + cIn.name + "' " + shapeToString (cIn.shape)
                      + " vs latent_dim=" + std::to_string (L) + ", cond_dim=" + std::to_string (C);
                return false;
            }

            im->layout = Impl::Layout::TwoInputs;
            im->zName  = zIn.name;   im->zShape = concreteShape (zIn.shape, L);
            im->cName  = cIn.name;   im->cShape = concreteShape (cIn.shape, C);
            im->zBuf.assign ((std::size_t) L, 0.0f);
            im->cBuf.assign ((std::size_t) C, 0.0f);

            io << im->zName << shapeToString (im->zShape) << " + " << im->cName << shapeToString (im->cShape);
        }
        else if (ins.size() == 1)
        {
            if (ins[0].perItem > 0 && ins[0].perItem != L + C)
            {
                error = "single input '" + ins[0].name + "' " + shapeToString (ins[0].shape)
                      + " is not latent_dim + cond_dim = " + std::to_string (L + C);
                return false;
            }
            im->layout   = Impl::Layout::Concatenated;
            im->catName  = ins[0].name;
            im->catShape = concreteShape (ins[0].shape, L + C);
            im->catBuf.assign ((std::size_t) (L + C), 0.0f);

            io << im->catName << shapeToString (im->catShape) << " (z then cond)";
        }
        else
        {
            error = "expected 1 or 2 model inputs, found " + std::to_string (ins.size());
            return false;
        }

        impl  = std::move (im);
        ready = true;

        // Smoke test so a broken model is caught at load time, not mid-performance.
        std::vector<float> z ((std::size_t) L, 0.0f), cond ((std::size_t) C, 0.0f), out;
        std::string decodeErr;
        if (! decode (z.data(), cond.data(), out, &decodeErr))
        {
            ready = false;
            impl.reset();
            error = "test decode failed: " + decodeErr;
            return false;
        }

        io << " -> " << lastOutputLength << " samples";
        if (lastOutputLength != (std::size_t) cfg.tableLen)
            io << " (resampled to " << cfg.tableLen << ")";
        summary = io.str();
        return true;
    }
    catch (const std::exception& e)
    {
        ready = false;
        impl.reset();
        error = e.what();
        return false;
    }
#else
    (void) modelFile;
    error = "built without ONNX Runtime (SIDVAE_WITH_ONNX=OFF)";
    return false;
#endif
}

bool SidVaeDecoder::decode (const float* z, const float* cond, std::vector<float>& out, std::string* error)
{
#if SIDVAE_WITH_ONNX
    if (! ready || impl == nullptr)
    {
        if (error) *error = "decoder not loaded";
        return false;
    }

    try
    {
        auto& im = *impl;
        const auto L = (std::size_t) cfg.latentDim;
        const auto C = (std::size_t) cfg.condDim;

        std::vector<Ort::Value>  inputs;
        std::vector<const char*> names;

        if (im.layout == Impl::Layout::TwoInputs)
        {
            std::copy (z,    z + L,    im.zBuf.begin());
            std::copy (cond, cond + C, im.cBuf.begin());
            inputs.push_back (Ort::Value::CreateTensor<float> (im.mem, im.zBuf.data(), L, im.zShape.data(), im.zShape.size()));
            inputs.push_back (Ort::Value::CreateTensor<float> (im.mem, im.cBuf.data(), C, im.cShape.data(), im.cShape.size()));
            names = { im.zName.c_str(), im.cName.c_str() };
        }
        else
        {
            std::copy (z,    z + L,    im.catBuf.begin());
            std::copy (cond, cond + C, im.catBuf.begin() + (std::ptrdiff_t) L);
            inputs.push_back (Ort::Value::CreateTensor<float> (im.mem, im.catBuf.data(), L + C, im.catShape.data(), im.catShape.size()));
            names = { im.catName.c_str() };
        }

        const char* outNames[] = { im.outName.c_str() };
        auto result = im.session->Run (Ort::RunOptions { nullptr },
                                       names.data(), inputs.data(), inputs.size(),
                                       outNames, 1);

        const std::size_t n = result[0].GetTensorTypeAndShapeInfo().GetElementCount();
        if (n < 2)
        {
            if (error) *error = "model output has fewer than 2 samples";
            return false;
        }
        lastOutputLength = n;

        const float* data = result[0].GetTensorData<float>();
        out.resize ((std::size_t) cfg.tableLen);
        if (n == out.size()) std::copy (data, data + n, out.begin());
        else                 resampleCyclic (data, n, out.data(), out.size());

        if (! allFinite (out.data(), out.size()))
        {
            if (error) *error = "model produced NaN/Inf";
            return false;
        }
        return true;
    }
    catch (const std::exception& e)
    {
        if (error) *error = e.what();
        return false;
    }
#else
    (void) z; (void) cond; (void) out;
    if (error) *error = "built without ONNX Runtime";
    return false;
#endif
}
}

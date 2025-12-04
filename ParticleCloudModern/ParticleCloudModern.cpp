// File: ParticleCloudModern.cpp
#define GL_SILENCE_DEPRECATION
#include <FFGL.h>
#include <FFGLLib.h>
#include <FFGLPluginSDK.h>
#include <FFGLLog.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

// Total de 46 parámetros
#define FFPARAM_COUNT             0
#define FFPARAM_SIZE              1
#define FFPARAM_LIFE              2
#define FFPARAM_EMIT_RATE         3
#define FFPARAM_VELOCITY          4
#define FFPARAM_GRAVITY           5
#define FFPARAM_FRICTION          6
#define FFPARAM_TURBULENCE        7
#define FFPARAM_MODE              8
#define FFPARAM_STRENGTH          9
#define FFPARAM_ATTRACT_X         10
#define FFPARAM_ATTRACT_Y         11
#define FFPARAM_COLOR_MODE        12
#define FFPARAM_HUE_START         13
#define FFPARAM_HUE_END           14
#define FFPARAM_SAT               15
#define FFPARAM_BRIGHT            16
#define FFPARAM_TRAIL_LENGTH      17
#define FFPARAM_TRAIL_FADE        18
#define FFPARAM_SHAPE_MODE        19
#define FFPARAM_FORM_STR          20
#define FFPARAM_CURL_SCALE        21
#define FFPARAM_CURL_SPEED        22
#define FFPARAM_FLOW_SCALE        23
#define FFPARAM_FLOW_SPEED        24
#define FFPARAM_ORBIT_SPEED       25
#define FFPARAM_ORBIT_RADIUS      26
#define FFPARAM_WAVE_AMP          27
#define FFPARAM_WAVE_FREQ         28
#define FFPARAM_SPIRAL_TIGHT      29
#define FFPARAM_SPIRAL_SPEED      30
#define FFPARAM_EXPLODE_FORCE     31
#define FFPARAM_RANDOM            32
#define FFPARAM_ALPHA             33
#define FFPARAM_BLEND             34
#define FFPARAM_GLOW              35
#define FFPARAM_TIME_SCALE        36
#define FFPARAM_RESET             37
#define FFPARAM_IMAGE_MASK        38
#define FFPARAM_MASK_POWER        39
#define FFPARAM_MASK_THRESHOLD    40
#define FFPARAM_SEED              41
#define FFPARAM_TOTAL             42

class ParticleCloudModern : public CFFGLPlugin
{
public:
    ParticleCloudModern();
    ~ParticleCloudModern();
    static FFResult __stdcall CreateInstance(CFFGLPlugin** ppOutInstance);
    FFResult InitGL(const FFGLViewportStruct *vp) override;
    FFResult ProcessOpenGL(ProcessOpenGLStruct *pGL) override;
    FFResult DeInitGL() override;
    FFResult SetFloatParameter(unsigned int dwIndex, float value) override;
    float GetFloatParameter(unsigned int dwIndex) override;

private:
    GLuint m_VAO, m_VBO, m_Program;
    float m_Params[FFPARAM_TOTAL];

    struct Particle {
        float x, y, vx, vy, life, age;
        float initialX, initialY; // Para Form Strength
    };

    struct VertexData {
        float x, y, r, g, b, a;
    };

    std::vector<Particle> m_Particles;
    std::vector<VertexData> m_VertexBuffer;

    unsigned int m_FrameCounter;
    void UpdateParticles(float dt);
    void HSVtoRGB(float h, float s, float v, float& r, float& g, float& b);
    void GetColor(float t, float& r, float& g, float& b);
    GLuint CompileShader(GLenum type, const char* source);
    void ApplyBehavior(Particle& p, int mode, float strength, float dt);
    void ApplyFlow(Particle& p, float dt);
    void ApplyCurl(Particle& p, float dt);
    void ApplySpiral(Particle& p, float dt);
    float Noise(float x, float y);
};

static CFFGLPluginInfo PluginInfo(
    ParticleCloudModern::CreateInstance,
    "PCLD",
    "Particle Cloud Ultimate",
    2, 1, 1, 0,
    FF_EFFECT,
    "Particles with full simulation and image-based masking",
    "Antigravity AI"
);

// Constructor
ParticleCloudModern::ParticleCloudModern() : CFFGLPlugin()
{
    SetMinInputs(1);
    SetMaxInputs(1);

    for (int i = 0; i < FFPARAM_TOTAL; i++) m_Params[i] = 0.5f;

    // Defaults
    m_Params[FFPARAM_COUNT] = 0.5f;
    m_Params[FFPARAM_SIZE] = 0.1f;
    m_Params[FFPARAM_LIFE] = 0.5f; 
    m_Params[FFPARAM_VELOCITY] = 0.2f;
    m_Params[FFPARAM_SAT] = 0.8f;
    m_Params[FFPARAM_BRIGHT] = 1.0f;
    m_Params[FFPARAM_ALPHA] = 0.9f;
    m_Params[FFPARAM_TIME_SCALE] = 0.5f;
    m_Params[FFPARAM_IMAGE_MASK] = 0.0f;
    m_Params[FFPARAM_MASK_POWER] = 0.5f;
    m_Params[FFPARAM_MASK_THRESHOLD] = 0.1f;
    m_Params[FFPARAM_BLEND] = 0.0f; 
    m_Params[FFPARAM_GLOW] = 0.0f;
    m_Params[FFPARAM_SEED] = 0.0f;
    m_Params[FFPARAM_TRAIL_LENGTH] = 0.0f;

    // Registro
    SetParamInfof(FFPARAM_COUNT, "Count", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_SIZE, "Size", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_LIFE, "Life", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_EMIT_RATE, "Emit Rate", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_VELOCITY, "Velocity", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_GRAVITY, "Gravity", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_FRICTION, "Friction", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_TURBULENCE, "Turbulence", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_MODE, "Behavior Mode", FF_TYPE_STANDARD); 
    SetParamInfof(FFPARAM_STRENGTH, "Strength", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_ATTRACT_X, "Attract X", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_ATTRACT_Y, "Attract Y", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_COLOR_MODE, "Color Mode", FF_TYPE_STANDARD); 
    SetParamInfof(FFPARAM_HUE_START, "Hue/Pal", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_HUE_END, "Hue End", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_SAT, "Saturation", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_BRIGHT, "Brightness", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_TRAIL_LENGTH, "Trail Length", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_TRAIL_FADE, "Trail Fade", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_SHAPE_MODE, "Shape Mode", FF_TYPE_STANDARD); 
    SetParamInfof(FFPARAM_FORM_STR, "Form Strength", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_CURL_SCALE, "Curl Scale", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_CURL_SPEED, "Curl Speed", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_FLOW_SCALE, "Flow Scale", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_FLOW_SPEED, "Flow Speed", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_ORBIT_SPEED, "Orbit Speed", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_ORBIT_RADIUS, "Orbit Radius", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_WAVE_AMP, "Wave Amp", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_WAVE_FREQ, "Wave Freq", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_SPIRAL_TIGHT, "Spiral Tight", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_SPIRAL_SPEED, "Spiral Speed", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_EXPLODE_FORCE, "Explode Force", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_RANDOM, "Randomness", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_ALPHA, "Alpha", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_BLEND, "Blend Mode", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_GLOW, "Glow", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_TIME_SCALE, "Time Scale", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_RESET, "Reset", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_IMAGE_MASK, "Image Mask", FF_TYPE_STANDARD); 
    SetParamInfof(FFPARAM_MASK_POWER, "Mask Power", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_MASK_THRESHOLD, "Mask Threshold", FF_TYPE_STANDARD);
    SetParamInfof(FFPARAM_SEED, "Random Seed", FF_TYPE_STANDARD);

    m_FrameCounter = 0;
}

ParticleCloudModern::~ParticleCloudModern() {}

// === SHADERS ===
const char* VERTEX_SHADER = R"(
    #version 410 core
    layout(location = 0) in vec2 aPos;
    layout(location = 1) in vec4 aColor;

    out vec4 vColor;
    out vec2 vTexCoord;

    uniform float uPointSize;

    void main() {
        gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0);
        gl_PointSize = uPointSize;
        vColor = aColor;
        vTexCoord = aPos;
    }
)";

const char* FRAGMENT_SHADER = R"(
    #version 410 core
    in vec4 vColor;
    in vec2 vTexCoord;
    out vec4 FragColor;

    uniform sampler2D uInputTexture;
    uniform float uImageMask;
    uniform float uMaskPower;
    uniform float uMaskThreshold;
    uniform float uShapeMode; // 0=Circle, 0.5=Square, 1.0=Star

    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);

        // Shape Logic
        if(uShapeMode < 0.33) { // Circle
            if(dist > 0.5) discard; 
        } else if(uShapeMode < 0.66) { // Square
            // No discard
        } else { // Star
            float angle = atan(coord.y, coord.x);
            float r = 0.5 * (0.5 + 0.5 * sin(5.0 * angle));
            if(dist > r + 0.1) discard;
        }

        // Image Mask Logic
        vec4 imgColor = texture(uInputTexture, vTexCoord);
        float brightness = dot(imgColor.rgb, vec3(0.299, 0.587, 0.114));
        brightness = pow(brightness, uMaskPower * 4.0 + 0.1); 
        
        float maskFactor = brightness * imgColor.a;

        if(uImageMask > 0.5) {
            if(maskFactor < uMaskThreshold) discard;
        }

        // Final Color
        float finalAlpha = vColor.a;
        if(uImageMask > 0.5) finalAlpha *= maskFactor;

        // Soft Edge for circles
        if(uShapeMode < 0.33) {
            float edge = 1.0 - smoothstep(0.4, 0.5, dist);
            finalAlpha *= edge;
        }

        FragColor = vec4(vColor.rgb, finalAlpha);
    }
)";

GLuint ParticleCloudModern::CompileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

FFResult ParticleCloudModern::InitGL(const FFGLViewportStruct *vp) {
    GLuint vs = CompileShader(GL_VERTEX_SHADER, VERTEX_SHADER);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    m_Program = glCreateProgram();
    glAttachShader(m_Program, vs);
    glAttachShader(m_Program, fs);
    glLinkProgram(m_Program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &m_VAO);
    glGenBuffers(1, &m_VBO);

    glBindVertexArray(m_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return FF_SUCCESS;
}

void ParticleCloudModern::UpdateParticles(float dt) {
    int maxParticles = 30000;
    int targetCount = static_cast<int>(m_Params[FFPARAM_COUNT] * maxParticles) + 100;
    
    // Trail Logic
    bool useTrails = m_Params[FFPARAM_TRAIL_LENGTH] > 0.01f;
    int vertexCount = useTrails ? targetCount * 2 : targetCount;

    if (m_Particles.size() != targetCount) {
        size_t oldSize = m_Particles.size();
        m_Particles.resize(targetCount);
        // Inicializar posiciones en grid para Form Strength
        int gridSide = (int)sqrt(targetCount);
        for(size_t i=0; i<targetCount; ++i) {
             m_Particles[i].life = 0.0f;
             float gx = (i % gridSide) / (float)gridSide;
             float gy = (i / gridSide) / (float)gridSide;
             m_Particles[i].initialX = gx;
             m_Particles[i].initialY = gy;
        }
    }
    
    if (m_VertexBuffer.size() != vertexCount) {
        m_VertexBuffer.resize(vertexCount);
    }

    float timeScale = m_Params[FFPARAM_TIME_SCALE] * 2.0f;
    float simDt = dt * timeScale;
    
    // Parámetros ampliados
    float lifeBase = m_Params[FFPARAM_LIFE] * 10.0f + 0.1f; // Hasta 10s
    float velocity = m_Params[FFPARAM_VELOCITY] * 0.5f;     // Más rápido
    float friction = 1.0f - (m_Params[FFPARAM_FRICTION] * 0.1f);
    float gravity = (m_Params[FFPARAM_GRAVITY] - 0.5f) * 0.1f;
    float strength = m_Params[FFPARAM_STRENGTH];
    float formStr = m_Params[FFPARAM_FORM_STR];
    float turbulence = m_Params[FFPARAM_TURBULENCE];
    float explodeForce = m_Params[FFPARAM_EXPLODE_FORCE] * 0.2f;
    float randomSeed = m_Params[FFPARAM_RANDOM] * 10000.0f;
    float trailLen = m_Params[FFPARAM_TRAIL_LENGTH] * 0.5f; 
    float trailFade = m_Params[FFPARAM_TRAIL_FADE];

    // Modos Discretos
    int mode = static_cast<int>(m_Params[FFPARAM_MODE] * 4.9f); // 0..4
    int colorMode = static_cast<int>(m_Params[FFPARAM_COLOR_MODE] * 2.9f); // 0..2

    srand(static_cast<unsigned int>(randomSeed + m_FrameCounter));

    for (size_t i = 0; i < m_Particles.size(); ++i) {
        Particle& p = m_Particles[i];
        p.age += simDt;

        if (p.age > p.life || m_Params[FFPARAM_RESET] > 0.5f) {
            float angle = ((float)rand() / RAND_MAX) * 6.283f;
            float radius = (float)rand() / RAND_MAX;
            p.x = 0.5f + cos(angle) * radius * 0.2f;
            p.y = 0.5f + sin(angle) * radius * 0.2f;
            p.vx = ((float)rand() / RAND_MAX - 0.5f) * velocity;
            p.vy = ((float)rand() / RAND_MAX - 0.5f) * velocity;
            p.life = lifeBase * (0.5f + 0.5f * ((float)rand() / RAND_MAX));
            p.age = 0.0f;
        }

        // Form Strength (Return to Origin)
        if (formStr > 0.01f) {
            float dx = p.initialX - p.x;
            float dy = p.initialY - p.y;
            p.vx += dx * formStr * simDt * 5.0f;
            p.vy += dy * formStr * simDt * 5.0f;
        }

        if (explodeForce > 0.001f && p.age < 0.2f) {
            float dx = p.x - 0.5f;
            float dy = p.y - 0.5f;
            p.vx += dx * explodeForce;
            p.vy += dy * explodeForce;
        }

        ApplyBehavior(p, mode, strength, simDt);
        ApplyFlow(p, simDt);
        ApplyCurl(p, simDt);
        ApplySpiral(p, simDt);

        p.vx += ((float)rand() / RAND_MAX - 0.5f) * 0.01f * turbulence;
        p.vy += ((float)rand() / RAND_MAX - 0.5f) * 0.01f * turbulence;

        p.vy += gravity;
        p.vx *= friction;
        p.vy *= friction;
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0) p.x += 1; if (p.x > 1) p.x -= 1;
        if (p.y < 0) p.y += 1; if (p.y > 1) p.y -= 1;

        // Color
        float r, g, b;
        float t = p.age / p.life;
        
        if (colorMode == 0) { // Single
             GetColor(m_Params[FFPARAM_HUE_START], r, g, b);
        } else if (colorMode == 1) { // Gradient
             float h = m_Params[FFPARAM_HUE_START] + (m_Params[FFPARAM_HUE_END] - m_Params[FFPARAM_HUE_START]) * t;
             GetColor(h, r, g, b);
        } else { // Palette
             int palIdx = (int)(m_Params[FFPARAM_HUE_START] * 5.9f);
             float offset = m_Params[FFPARAM_HUE_END] + t;
             float ph = 0.0f;
             if(palIdx == 0) ph = offset; // Rainbow
             if(palIdx == 1) ph = 0.0f + offset * 0.1f; // Reds
             if(palIdx == 2) ph = 0.6f + offset * 0.2f; // Blues
             if(palIdx == 3) ph = 0.3f + offset * 0.2f; // Greens
             if(palIdx == 4) ph = (int)(offset * 4.0f) * 0.25f; // Discrete
             GetColor(ph, r, g, b);
        }

        // FADE IN / FADE OUT
        // Fade in durante el primer 10% de vida (t < 0.1)
        float x = (t < 0.1f) ? (t * 10.0f) : 1.0f; // Normalizar 0..1 en el rango 0..0.1
        float fadeIn = x * x * (3.0f - 2.0f * x); // Fórmula Smoothstep manual
        
        float fadeOut = 1.0f - t;
        float alpha = fadeIn * fadeOut * m_Params[FFPARAM_ALPHA];

        // Fill Vertex Buffer
        if (useTrails) {
            m_VertexBuffer[i*2] = { p.x, p.y, r, g, b, alpha };
            float tailX = p.x - p.vx * trailLen * 10.0f;
            float tailY = p.y - p.vy * trailLen * 10.0f;
            float tailAlpha = alpha * (1.0f - trailFade);
            m_VertexBuffer[i*2+1] = { tailX, tailY, r, g, b, tailAlpha };
        } else {
            m_VertexBuffer[i] = { p.x, p.y, r, g, b, alpha };
        }
    }
    m_FrameCounter++;
}

void ParticleCloudModern::GetColor(float h, float& r, float& g, float& b) {
    HSVtoRGB(h, m_Params[FFPARAM_SAT], m_Params[FFPARAM_BRIGHT], r, g, b);
}

void ParticleCloudModern::ApplyBehavior(Particle& p, int mode, float strength, float dt) {
    float ax = m_Params[FFPARAM_ATTRACT_X];
    float ay = m_Params[FFPARAM_ATTRACT_Y];
    float dx = ax - p.x;
    float dy = ay - p.y;
    float dist = sqrtf(dx * dx + dy * dy) + 0.0001f;

    switch (mode) {
        case 0: // Random
            p.vx += ((float)rand() / RAND_MAX - 0.5f) * strength * 0.01f;
            p.vy += ((float)rand() / RAND_MAX - 0.5f) * strength * 0.01f;
            break;
        case 1: // Attract
            p.vx += (dx / dist) * strength * dt * 5.0f;
            p.vy += (dy / dist) * strength * dt * 5.0f;
            break;
        case 2: // Repel
            p.vx -= (dx / dist) * strength * dt * 5.0f;
            p.vy -= (dy / dist) * strength * dt * 5.0f;
            break;
        case 3: // Orbit
            p.vx += -dy * strength * dt * 5.0f;
            p.vy += dx * strength * dt * 5.0f;
            break;
        case 4: // Wave
            p.vy += sinf(p.x * m_Params[FFPARAM_WAVE_FREQ] * 10.0f + m_FrameCounter * 0.1f) * m_Params[FFPARAM_WAVE_AMP] * 0.1f;
            break;
    }
}

void ParticleCloudModern::ApplySpiral(Particle& p, float dt) {
    float dx = p.x - 0.5f;
    float dy = p.y - 0.5f;
    float tightness = m_Params[FFPARAM_SPIRAL_TIGHT] * 10.0f;
    float speed = m_Params[FFPARAM_SPIRAL_SPEED] * 5.0f;
    p.vx += -dy * tightness * dt * speed;
    p.vy += dx * tightness * dt * speed;
}

void ParticleCloudModern::ApplyFlow(Particle& p, float dt) {
    float scale = m_Params[FFPARAM_FLOW_SCALE] * 10.0f;
    float speed = m_Params[FFPARAM_FLOW_SPEED] * 0.1f;
    float angle = Noise(p.x * scale, p.y * scale + m_FrameCounter * speed) * 6.283f;
    p.vx += cosf(angle) * dt * 0.01f;
    p.vy += sinf(angle) * dt * 0.01f;
}

void ParticleCloudModern::ApplyCurl(Particle& p, float dt) {
    float scale = m_Params[FFPARAM_CURL_SCALE] * 10.0f;
    float speed = m_Params[FFPARAM_CURL_SPEED] * 0.1f;
    float eps = 0.01f;
    float n1 = Noise(p.x + eps, p.y);
    float n2 = Noise(p.x - eps, p.y);
    float dx = (n1 - n2) / (2.0f * eps);
    n1 = Noise(p.x, p.y + eps);
    n2 = Noise(p.x, p.y - eps);
    float dy = (n1 - n2) / (2.0f * eps);
    p.vx += dy * scale * dt * speed;
    p.vy -= dx * scale * dt * speed;
}

void ParticleCloudModern::HSVtoRGB(float h, float s, float v, float& r, float& g, float& b) {
    h -= floor(h);
    h *= 6.0f;
    int i = static_cast<int>(floor(h));
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
}

float ParticleCloudModern::Noise(float x, float y) {
    int n = (int)(x * 137 + y * 149);
    n = (n << 13) ^ n;
    return 1.0f - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0f;
}

FFResult ParticleCloudModern::ProcessOpenGL(ProcessOpenGLStruct *pGL) {
    // CRITICAL FIX: Disable Depth Test to prevent flickering
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    UpdateParticles(0.016f);

    glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
    glBufferData(GL_ARRAY_BUFFER, m_VertexBuffer.size() * sizeof(VertexData), m_VertexBuffer.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glUseProgram(m_Program);

    // Texture
    if (pGL->numInputTextures > 0 && pGL->inputTextures[0] != nullptr) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, pGL->inputTextures[0]->Handle);
        glUniform1i(glGetUniformLocation(m_Program, "uInputTexture"), 0);
    }

    // Uniforms
    glUniform1f(glGetUniformLocation(m_Program, "uPointSize"), m_Params[FFPARAM_SIZE] * 100.0f + 1.0f); 
    
    // Binary Mask Logic
    float maskVal = m_Params[FFPARAM_IMAGE_MASK] > 0.5f ? 1.0f : 0.0f;
    glUniform1f(glGetUniformLocation(m_Program, "uImageMask"), maskVal);
    
    glUniform1f(glGetUniformLocation(m_Program, "uMaskPower"), m_Params[FFPARAM_MASK_POWER]);
    glUniform1f(glGetUniformLocation(m_Program, "uMaskThreshold"), m_Params[FFPARAM_MASK_THRESHOLD]);
    glUniform1f(glGetUniformLocation(m_Program, "uShapeMode"), m_Params[FFPARAM_SHAPE_MODE]);

    // Blend
    glEnable(GL_BLEND);
    int blendMode = (int)(m_Params[FFPARAM_BLEND] * 3.99f);
    switch (blendMode) {
        case 0: glBlendFunc(GL_SRC_ALPHA, GL_ONE); break;
        case 1: glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); break;
        case 2: glBlendFunc(GL_DST_COLOR, GL_ZERO); break;
        case 3: glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR); break;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
    glBindVertexArray(m_VAO);

    // Draw
    if (m_Params[FFPARAM_TRAIL_LENGTH] > 0.01f) {
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_VertexBuffer.size()));
    } else {
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(m_VertexBuffer.size()));
    }

    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glUseProgram(0);

    return FF_SUCCESS;
}

FFResult ParticleCloudModern::DeInitGL() {
    if (m_Program) glDeleteProgram(m_Program);
    if (m_VBO) glDeleteBuffers(1, &m_VBO);
    if (m_VAO) glDeleteVertexArrays(1, &m_VAO);
    return FF_SUCCESS;
}
FFResult ParticleCloudModern::SetFloatParameter(unsigned int dwIndex, float value) {
    if (dwIndex < FFPARAM_TOTAL) m_Params[dwIndex] = value;
    return FF_SUCCESS;
}
float ParticleCloudModern::GetFloatParameter(unsigned int dwIndex) {
    if (dwIndex < FFPARAM_TOTAL) return m_Params[dwIndex];
    return 0.0f;
}
FFResult __stdcall ParticleCloudModern::CreateInstance(CFFGLPlugin** ppOutInstance) {
    *ppOutInstance = new ParticleCloudModern();
    return (*ppOutInstance != nullptr) ? FF_SUCCESS : FF_FAIL;
}
/ /   t r i g g e r  
 
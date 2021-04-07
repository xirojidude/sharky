Shader "Unlit/HUD2"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

       Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }



#define PI 3.14159265359
#define QUARTER_PI 0.78539816339
#define HALF_PI 1.57079632679
#define TWO_PI 6.28318530718

const float4  BLUE            = float4(0.306, 0.502, 0.537, 1.000);
const float4  RED             = float4(0.941, 0.306, 0.208, 1.000);
const float4  GREY_BLUE       = float4(0.494, 0.620, 0.663, 1.000);
const float4  YELLOW          = float4(0.969, 1.000, 0.804, 1.000);
const float4  GREEN           = float4(0.804, 1.000, 0.965, 1.000);
const float4  GREY            = float4(0.449, 0.481, 0.489, 1.000);
const float4  D_GREY          = float4(0.050, 0.050, 0.050, 1.000);
const float4  M_GREY          = float4(0.200, 0.200, 0.200, 1.000);
const float4  WHITE           = float4(1.000, 1.000, 1.000, 1.000);
const float4  T_WHITE         = float4(1.000, 1.000, 1.000, 0.000);
const float4  BLACK           = float4(0.000, 0.000, 0.000, 0.000);
const float2  ORIGIN          = float2(0.0,0);
const float LINE_WEIGHT     = 0.035;
const float METER_WEIGHT    = 0.100;
      float SMOOTH          = 0.0;
      float R_SMOOTH        = 0.2000;

float map(float value, float istart, float istop, float ostart, float ostop) 
{
    return ostart + (ostop - ostart) * ((value - istart) / (istop - istart));
}


float n_angleBetween(float2 v1, float2 v2)
{
    float angle = atan2( v1.x-v2.x,v1.y-v2.y);
    return ((angle < 0.0) ? (angle += TWO_PI) : angle) / TWO_PI;
}

float2 rotate(float2 v, float rotation)
{
    return float2(cos(rotation)*v.x + sin(rotation)*v.y, -sin(rotation)*v.x + cos(rotation)*v.y);
}

float radialLine(float2 r, float radius, float cutoff, float weight, bool flip)
{
    float2  uv = rotate(r, -cutoff/2.0);
    if(flip) uv = float2(0.0,0)-uv;
    float d = distance(uv, ORIGIN);
    float ir =  radius;
    float or =  ir + weight;
    float theta = n_angleBetween(uv, ORIGIN);
    float s = theta * 180.0;
    float line1 =  step(theta, cutoff/TWO_PI);
    float ring =  smoothstep(or+SMOOTH, or-SMOOTH, d) * (1.0 - smoothstep(ir+SMOOTH, ir-SMOOTH, d));
    
    return ring * line1;;
}

float radialMeter(float2 r, float radius, float cutoff, bool flip)
{
    float2  uv = rotate(r, -cutoff/2.0);
    if(flip) uv = float2(0.0,0)-uv;
    float d = distance(uv, ORIGIN);
    float ir =  radius;  
    float or =  ir + METER_WEIGHT;   
    float theta = n_angleBetween(uv, ORIGIN);
    float s = theta * 200.0;
    float grads = frac(s);    
    float steps = (1.0-smoothstep((0.4-R_SMOOTH), (0.41+R_SMOOTH), grads)) * smoothstep((0.2-R_SMOOTH),(0.21+R_SMOOTH), grads) ;
    float line1 =  step(theta, cutoff/TWO_PI);
    float ring =  smoothstep(or+SMOOTH, or-SMOOTH, d) * (1.0 - smoothstep(ir+SMOOTH, ir-SMOOTH, d));

    steps *= ring * line1;
        
    return steps;
}

float dottedSect(float2 r, float radius, float cutoff, float dotSize, bool flip, float num)
{ 
    float a = num;
    float2  uv = rotate(r, -cutoff/2.0);
    if(flip) uv = float2(0.0,0)-uv;
    float theta =  round(n_angleBetween(uv, float2(0.0,0)) * (TWO_PI * a)) / a;
    float x = radius * cos(theta);
    float y = radius * sin(theta);
    float2  v = float2(x, y);   
    float d = distance(uv, v);
    float e = smoothstep(d-SMOOTH, d+SMOOTH, dotSize);
    float line1 =  step(theta, cutoff);

    return e*line1;
}

float returnBracket(float2 r, float radius, float cutoff, float weight, float vCut, bool flip)
{
    float2 uv =  rotate(r, -cutoff/2.0);
    if(flip) uv = float2(0.0,0)-uv;
    float d = distance(uv, ORIGIN);
    float ir =  radius;
    float or =  ir + weight;
    float theta = n_angleBetween(uv, ORIGIN);
    float line1 =   step(theta, cutoff/TWO_PI);
    float halfs =  (cutoff/TWO_PI) / 2.0;
    float sec = step(halfs-(vCut/TWO_PI), theta) * (1.0-step(halfs+((vCut/TWO_PI)), theta));
    float ring = smoothstep(or+SMOOTH, or-SMOOTH, d) * (1.0 - smoothstep(ir+SMOOTH, ir-SMOOTH, d));
  
    return ring * (line1 - sec); 
}

float solBracket(float2 r, float radius, float weight, float cutoff, float cut, float retrn)
{
    float2 uv =  r;
    float d = distance(uv, ORIGIN);
    float ir =  radius;
    float or =  ir + weight;
    float ring =  smoothstep(ir-SMOOTH, ir+SMOOTH, d) * (1.0 - smoothstep(or-SMOOTH, or+SMOOTH, d));
    float circ = step(radius+weight, d);
    float block =   smoothstep(uv.y-SMOOTH, uv.y+SMOOTH, cutoff) * (1.0 - smoothstep(uv.y-SMOOTH, uv.y+SMOOTH, -cutoff));
    float topLine = smoothstep(uv.y-SMOOTH, uv.y+SMOOTH,  cutoff-weight);
    float botLine = smoothstep(uv.y-SMOOTH, uv.y+SMOOTH, -(cutoff-(weight)));
    
    float xblock = step(-retrn, uv.x) * (1.0-step(retrn, uv.x));
    float yblock = step(-cut, uv.y) * (1.0-step(cut, uv.y));
    
    ring *= block;
    topLine *= (1.0 - botLine);
    block *= (1.0 - topLine);
    ring += block;
    ring *= 1.0-circ;
    float blocks = (xblock+yblock);
    
    return ring * (1.0-(blocks));   
}

float single(float2 r, float radius, float weight)
{
    float2  uv = r;
    float d = distance(uv, ORIGIN);
    float ir =  radius;
    float or =  ir + weight;
    float ring =  smoothstep(or+SMOOTH, or-SMOOTH, d) * (1.0 - smoothstep(ir+SMOOTH, ir-SMOOTH, d));
    
    return ring;
}

float splitLine(float2 r, float w, float yOffset, float split, float weight) 
{
    float2 uv = r + float2(0.0, yOffset);
    float f = step(uv.x, w) * (1.0-step(w, -uv.x));
    float l = step(uv.y, (weight/2.0)) * (1.0 - step(uv.y, -(weight/2.0)));
    float g = step(uv.x, split) * (1.0-step(split, -uv.x));
    
    return f * l * (1.0-g);
}

float meter(float2 r, float w, float yOffset, float inc, float weight, float num) 
{
    float2 uv = r - float2(0.0, yOffset);
    float f = step(uv.x, w) * (1.0-step(w, -uv.x));
    float l = step(uv.y, (weight/2.0)) * (1.0 - step(uv.y, -(weight/2.0)));
    float incr = frac(uv.x*num);
    float gnn = (1.0-smoothstep((0.4-SMOOTH), (0.41+SMOOTH), incr)) * smoothstep((0.2-SMOOTH),(0.21+SMOOTH), incr);
    f *= (gnn * l);   
    
    return f ;
}

float grid(float2 r, float num, float weight)
{
    float2 uv = r * num;
    float gridx = smoothstep(weight - SMOOTH, weight + SMOOTH, frac(uv.x));
    float gridy = smoothstep(weight - SMOOTH, weight + SMOOTH, frac(uv.y));
    
    return (1.0 - (gridx * gridy));
}

float dots(float2 r, float num, float pointSize)
{
    float2 uv = r * num;
    float2  v = float2(round(uv.x),round(uv.y));   
    float d = distance(uv, v);
    return smoothstep(d-SMOOTH, d+SMOOTH, pointSize);   
}

float bg(float2 r, float w, float h)
{
    float f = 1.0 - step(w, distance(r, ORIGIN));
    
    float g = step(-h, r.y) * (1.0-step(h, r.y));
    
    f *= g;
    
    return 1.0-f;
}

float hash(float x)
{
    return frac(sin(x) * 43758.5453) * 2.0 - 1.0;
}

float2 hashPosition(float x)
{
    return float2(hash(x) * 64.0, hash(x * 1.1) * 64.0);
}

float sineOut(float t) 
{
  return sin(t * HALF_PI);
}

            fixed4 frag (v2f fragCoord) : SV_Target
            {
                // sample the texture
                fixed4 fragColor = tex2D(_MainTex, fragCoord.uv);

    SMOOTH      = map(800, 800.0, 2560.0, 0.0025, 0.0010);      //map(iResolution.x, 800.0, 2560.0, 0.0025, 0.0010);
    float2 uv =  2.0 * float2(fragCoord.uv - 0.5); // * float2(800,450)) / 450;
    
    float rotation = - PI / 4.0;
    rotation = 0.0;
    uv = float2(cos(rotation)*uv.x + sin(rotation)*uv.y, -sin(rotation)*uv.x + cos(rotation)*uv.y);
    
    float2 q = fragCoord.uv ; //   / iResolution.xy;
    
    const float sequenceDuration = 1.25;
    float currentSequence = floor(_Time.y / sequenceDuration);
    float currentSequenceTime = ((_Time.y)% sequenceDuration);
    float2  startingPosition = hashPosition(currentSequence) * 0.005;
    float2  goalPosition = hashPosition(currentSequence + 1.0) * 0.005;
    float2  currentPosition;
    const float speed = 0.5;
    float potentialDistance = speed * currentSequenceTime;
    float goalDistance = length(goalPosition - startingPosition);
    
    if (potentialDistance < goalDistance) {
        currentPosition = lerp(startingPosition, goalPosition, sineOut(potentialDistance / goalDistance));
    } else {
        currentPosition = goalPosition;
    }
    
    float2 targetPosition = uv-currentPosition;
    
    float4 tex = float4(0.,0.,0.,1);//*.5; //texture(iChannel0, q) * 0.1; 
    
    float4 final   = tex; //lerp(tex,  BLACK,  bg(uv, 1.4805, 0.500) * 0.50);      
         final *= lerp(tex,  BLACK,  bg(uv, 1.2805, 0.702) * 0.25); 
    
    final = lerp(final,  WHITE   , dots(uv, 20.0, 0.04) * 0.25);    
    final = lerp(final,  WHITE   , grid (uv, 10.0, 0.03) * 0.10);   
    final = lerp(final,  RED     , radialLine (targetPosition, 0.1092, QUARTER_PI, LINE_WEIGHT, false));    
    final = lerp(final,  RED     , radialLine (targetPosition, 0.1092, QUARTER_PI, LINE_WEIGHT, true)); 
    final = lerp(final,  WHITE   , radialLine (targetPosition, 0.2777, QUARTER_PI, LINE_WEIGHT, false));    
    final = lerp(final,  WHITE   , radialLine (targetPosition, 0.2777, QUARTER_PI, LINE_WEIGHT, true));
    final = lerp(final,  BLUE    , splitLine  (targetPosition, 0.3000, 0.3300, 0.0160, 0.005));
    final = lerp(final,  WHITE   , splitLine  (targetPosition, 0.6231, 0.0000, 0.5324, LINE_WEIGHT));
    final = lerp(final,  RED     , dottedSect (targetPosition, 0.3490, HALF_PI  +(QUARTER_PI/4.0), 0.004, false, 12.40)); 
    final = lerp(final,  RED     , dottedSect (targetPosition, 0.3490, HALF_PI  +(QUARTER_PI/4.0), 0.004, true, 12.40));  
    final = lerp(final,  GREY    , solBracket (targetPosition, 0.3490, LINE_WEIGHT, 0.3, 0.265, 0.000));
    final = lerp(final,  WHITE   , solBracket (targetPosition, 0.3777, LINE_WEIGHT, 0.05, 0.018, 0.3675));
    final = lerp(final,  WHITE   , radialMeter(targetPosition, 0.4138, QUARTER_PI, false));   
    final = lerp(final,  WHITE   , radialMeter(targetPosition, 0.4138, QUARTER_PI, true)); 
    final = lerp(final,  GREY    , single     (targetPosition, 0.4527, LINE_WEIGHT));
    final = lerp(final,  GREY_BLUE, splitLine  (targetPosition, 0.6231, 0.0, 0.5324, LINE_WEIGHT));    
    final = lerp(final,  BLUE    , solBracket (targetPosition, 0.5750, LINE_WEIGHT, 1.0, 0.2314, 0.0000));
    final = lerp(final,  WHITE   , solBracket (uv, 1.4805, LINE_WEIGHT, 0.7000, 0.0592, 1.2900));
    final = lerp(final,  BLUE    , solBracket (uv, 1.3021, LINE_WEIGHT, 0.6685, 0.1592, 0.0000));
    final = lerp(final,  GREY    , solBracket (uv, 1.3021, LINE_WEIGHT, 0.7000, 0.6950, 0.0000));
    final = lerp(final,  BLUE    , solBracket (uv, 1.2574, LINE_WEIGHT, 0.5174, 0.0000, 0.8551));
    final = lerp(final,  BLUE    , solBracket (uv, 1.2574, LINE_WEIGHT, 0.5374, 0.5300, 0.8551));
    final = lerp(final,  BLUE    , solBracket (uv, 1.2574, LINE_WEIGHT, 0.6390, 0.6300, 0.8551));
    final = lerp(final,  WHITE   , dottedSect (uv, 1.2800, 0.8324, 0.003, false, 112.40));
    final = lerp(final,  WHITE   , dottedSect (uv, 1.2800, 0.8324, 0.003, true, 112.40));
    final = lerp(final,  GREY    , meter      (uv, 0.8500, 0.6390, 0.9324, 0.010, 50.0));

   
    

    fragColor = final;

                return fragColor;
            }
            ENDCG
        }
    }
}


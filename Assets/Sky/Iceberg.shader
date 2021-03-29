
Shader "Skybox/Iceberg"

{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
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
            #pragma exclude_renderers d3d11_9x
            #pragma exclude_renderers d3d9

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }


/*--------------------------------------------------------------------------------------
License CC0 - http://creatifloatommons.org/publicdomain/zero/1.0/
To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty.
----------------------------------------------------------------------------------------
^ This means do ANYTHING YOU WANT with this code. Because we are programmers, not lawyers.
-Otavio Good
*/

// ---------------- Config ----------------
// This is an option that lets you render high quality frames for screenshots. It enables
// stochastic antialiasing and motion blur automatically for any shader.
//#define NON_REALTIME_HQ_RENDER
const float frameToRenderHQ = 15.1; // Time in seconds of frame to render
const float antialiasingSamples = 16.0; // 16x antialiasing - too much might make the shader compiler angry.

#define MANUAL_CAMERA
// Some computers were crashing, so I scaled this down by default.
//#define HQ_NOISE

// Makes compile times much faster.
// Forces for loops to not unroll because compiler thinks the zero is not a constant.
#define ZERO_TRICK max(0, _WorldSpaceCameraPos.x)

// --------------------------------------------------------
// These variables are for the non-realtime block renderer.
float localTime = 0.0;
float seed = 1.0;

// Animation variables
float fade = 1.0;
float exposure = 1.0;

// lighting vars
float3 sunDir = normalize(float3(0.93, 1.0, 1.0));
const float3 sunCol = float3(250.0, 220.0, 200.0) / 3555.0;
const float3 horizonCol = float3(0.95, 0.95, 0.95)*1.3;
const float3 skyCol = float3(0.03,0.45,0.95);
const float3 groundCol = float3(0.003,0.7,0.75);

// ---- noise functions ----
float v31(float3 a)
{
    return a.x + a.y * 37.0 + a.z * 521.0;
}
float v21(float2 a)
{
    return a.x + a.y * 37.0;
}
float Hash11(float a)
{
    return frac(sin(a)*10403.9);
}
float Hash21(float2 uv)
{
    float f = uv.x + uv.y * 37.0;
    return frac(sin(f)*104003.9);
}
float2 Hash22(float2 uv)
{
    float f = uv.x + uv.y * 37.0;
    return frac(cos(f)*float2(10003.579, 37049.7));
}
float2 Hash12(float f)
{
    return frac(cos(f)*float2(10003.579, 37049.7));
}
// noise functions
float Hash2d(float2 uv)
{
    float f = uv.x + uv.y * 37.0;
    return frac(sin(f)*104003.9);
}
float Hash3d(float3 uv)
{
    float f = uv.x + uv.y * 37.0 + uv.z * 521.0;
    return frac(sin(f)*110003.9);
}
float lerpP(float f0, float f1, float a)
{
    return lerp(f0, f1, a*a*(3.0-2.0*a));
}
const float2 zeroOne = float2(0.0, 1.0);
float noise1d(float uv)
{
    float fr = frac(uv);
    float fl = floor(uv);
    float h0 = Hash11(fl);
    float h1 = Hash11(fl + 1.0);
    return lerpP(h0, h1, fr);
}
float noise2d(float2 uv)
{
    float2 fr = frac(uv.xy);
    float2 fl = floor(uv.xy);
    float h00 = Hash2d(fl);
    float h10 = Hash2d(fl + zeroOne.yx);
    float h01 = Hash2d(fl + zeroOne);
    float h11 = Hash2d(fl + zeroOne.yy);
    return lerpP(lerpP(h00, h10, fr.x), lerpP(h01, h11, fr.x), fr.y);
}
float noiseValue(float3 uv)
{
    float3 fr = frac(uv.xyz);
    float3 fl = floor(uv.xyz);
    float h000 = Hash3d(fl);
    float h100 = Hash3d(fl + zeroOne.yxx);
    float h010 = Hash3d(fl + zeroOne.xyx);
    float h110 = Hash3d(fl + zeroOne.yyx);
    float h001 = Hash3d(fl + zeroOne.xxy);
    float h101 = Hash3d(fl + zeroOne.yxy);
    float h011 = Hash3d(fl + zeroOne.xyy);
    float h111 = Hash3d(fl + zeroOne.yyy);
    return lerpP(
        lerpP(lerpP(h000, h100, fr.x),
             lerpP(h010, h110, fr.x), fr.y),
        lerpP(lerpP(h001, h101, fr.x),
             lerpP(h011, h111, fr.x), fr.y)
        , fr.z);
}


const float PI=3.14159265;

float3 saturate(float3 a) { return clamp(a, 0.0, 1.0); }
float2 saturate(float2 a) { return clamp(a, 0.0, 1.0); }
float saturate(float a) { return clamp(a, 0.0, 1.0); }

float3 RotateX(float3 v, float rad)
{
  float cos1 = cos(rad);
  float sin1 = sin(rad);
  return float3(v.x, cos1 * v.y + sin1 * v.z, -sin1 * v.y + cos1 * v.z);
}
float3 RotateY(float3 v, float rad)
{
  float cos1 = cos(rad);
  float sin1 = sin(rad);
  return float3(cos1 * v.x - sin1 * v.z, v.y, sin1 * v.x + cos1 * v.z);
}
float3 RotateZ(float3 v, float rad)
{
  float cos1 = cos(rad);
  float sin1 = sin(rad);
  return float3(cos1 * v.x + sin1 * v.y, -sin1 * v.x + cos1 * v.y, v.z);
}

// This function basically is a procedural environment map that makes the sun
float3 GetSunColorSmall(float3 rayDir, float3 sunDir)
{
    float3 localRay = normalize(rayDir);
    float dist = 1.0 - (dot(localRay, sunDir) * 0.5 + 0.5);
    float sunIntensity = 0.05 / dist;
    sunIntensity += exp(-dist*150.0)*7000.0;
    sunIntensity = min(sunIntensity, 40000.0);
    return sunCol * sunIntensity*0.2;
}

float3 GetEnvMap(float3 rayDir, float3 sunDir)
{
    // fade the sky color, multiply sunset dimming
    float3 finalColor = lerp(horizonCol, skyCol, pow(saturate(rayDir.y), 0.47))*0.95;
    // make clouds - just a horizontal plane with noise
    float n = noise2d(rayDir.xz/rayDir.y*1.0);
    n += noise2d(rayDir.xz/rayDir.y*2.0)*0.5;
    n += noise2d(rayDir.xz/rayDir.y*4.0)*0.25;
    n += noise2d(rayDir.xz/rayDir.y*8.0)*0.125;
    n = pow(abs(n), 3.0);
    n = lerp(n * 0.2, n, saturate(abs(rayDir.y * 8.0)));  // fade clouds in distance
    finalColor = lerp(finalColor, (float3(1.0,1,1)+sunCol*10.0)*0.75*saturate((rayDir.y+0.2)*5.0), saturate(n*0.125));

    // add the sun
    finalColor += GetSunColorSmall(rayDir, sunDir);
    return finalColor;
}

// min function that supports materials in the y component
float2 matmin(float2 a, float2 b)
{
    if (a.x < b.x) return a;
    else return b;
}

// ---- shapes defined by distance fields ----
// See this site for a reference to more distance functions...
// http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

// signed box distance field
float sdBox(float3 p, float3 radius)
{
  float3 dist = abs(p) - radius;
  return min(max(dist.x, max(dist.y, dist.z)), 0.0) + length(max(dist, 0.0));
}

float cyl(float3 p, float rad)
{
    return length(p.xy) - rad;
}

float sSphere(float3 p, float rad)
{
    return length(p) - rad;
}

// k should be negative. -4.0 works nicely.
// smooth blending function
float smin(float a, float b, float k)
{
    return log2(exp2(k*a)+exp2(k*b))/k;
}

const float sway = 0.015;

// This is the distance function that defines all the scene's geometry.
// The input is a position in space.
// The output is the distance to the nearest surface, a material index,
// and the difference between the iceberg distance and the water distance.
float3 DistanceToObject(float3 p)
{
    float dist = p.y;

    if (abs(dist) < 0.07)  // Only calculate noise if we are close.
    {
        // Sum up different frequencies of noise to make water waves.
        float waterNoise = noise2d(p.xz*4.0+localTime)*0.1 +
            noise2d(p.xz*8.0+localTime)*0.03 +
            noise2d(p.xz*16.0-localTime)*0.015 +
            noise2d(p.xz*32.0-localTime)*0.005 +
            noise2d(p.xz*64.0-localTime)*0.002;
        // Fade the waves a bit in the distance.
        dist += waterNoise * 0.2 * saturate(8.0/length(p.xz));
    }
    float2 water = float2(dist, 1.0);

    p = RotateZ(p, sin(localTime)*sway);
    // Sculpt the iceberg.
    float slant = (p.y + p.x*0.25) / 1.0307;
    slant -= cos(p.z*2.0)*0.0625;
    dist = sSphere(p, 2.0) + sin(p.z*4.0)*0.125;
    dist = smin(dist, sSphere(p + float3(1.0, 11.85, 0.0), 12.0), -5.0);
    float chop = cyl(p.xzy + float3(1.5,1.5,1.5), 1.5);
    float chop2 = cyl(p.xyz + float3(0.0, -0.5, 0.0), 0.6) + sin(p.z*2.0)*0.125;
    chop2 = min(chop2, -slant + 1.6);
    chop2 = min(chop2, sdBox(p + float3(-1.75, -0.74, -2.0), float3(0.7,.7,.7)));
    chop = smin(chop, chop2, -10.0);
    chop = min(chop, chop2);
    dist = -smin(-dist, chop, -30.0);
    if (abs(dist) < 0.5)  // Only calculate noise if we are close.
    {
        //dist += noise1d(slant*4.0+1.333)*0.1 + noise1d(slant*8.0+1.333)*0.05;
        dist += noiseValue(float3(slant,slant,slant)*4.0)*0.1 + noiseValue(float3(slant,slant,slant)*8.0)*0.05;
        float snowNoise=0.0;
        snowNoise = noiseValue(p*4.0)*0.5*0.5;
        snowNoise += noiseValue(p*8.0)*0.125*0.25;
        // prevent crashing on mac/chrome/nvidia
#ifdef HQ_NOISE
        snowNoise += noiseValue(p*16.0)*0.125*0.0625;
        snowNoise += noiseValue(p*32.0)*0.0625*0.0625;
#endif
        //snowNoise -= abs(frac(p.z*0.5-p.y*0.05)-0.5)*2.0;
        //snowNoise -= 0.95;
        dist += snowNoise*0.25;
    }
    float2 iceberg = float2(dist, 0.0);
    float2 distAndMat = matmin(water, iceberg);
    return float3(distAndMat, water.x - iceberg.x);
}

float3 TraceOneRay(float3 camPos, float3 rayfloat, out float3 normal, out float3 distAndMat, out float t) {
    normal = float3(0.0,0,0);
    distAndMat = float3(0.0, -1.0, 1000.0);  // Distance and material
    float3 finalColor = float3(0.0,0,0);
    // ----------------------------- Ray march the scene ------------------------------
    t = 0.0;
    const float maxDepth = 32.0; // farthest distance rays will travel
    float3 pos = float3(0.0,0,0);
    const float smallVal = 0.00625;
    // ray marching time
    for (int i = 210; i >= ZERO_TRICK; i--) // This is the count of the max times the ray actually marches.
    {
        // Step along the ray.
        pos = (camPos + rayfloat * t);
        // This is _the_ function that defines the "distance field".
        // It's really what makes the scene geometry. The idea is that the
        // distance field returns the distance to the closest object, and then
        // we know we are safe to "march" along the ray by that much distance
        // without hitting anything. We repeat this until we get really close
        // and then break because we have effectively hit the object.
        distAndMat = DistanceToObject(pos);

        // move down the ray a safe amount
        t += distAndMat.x;
        if (i == 0) t = maxDepth+0.01;
        // If we are very close to the object, let's call it a hit and exit this loop.
        if ((t > maxDepth) || (abs(distAndMat.x) < smallVal)) break;
    }

    // --------------------------------------------------------------------------------
    // Now that we have done our ray marching, let's put some color on this geometry.
    // If a ray actually hit the object, let's light it.
    if (t <= maxDepth)
    {
        float dist = distAndMat.x;
        // calculate the normal from the distance field. The distance field is a volume, so if you
        // sample the current point and neighboring points, you can use the difference to get
        // the normal.
        float3 smallfloat = float3(smallVal, 0, 0);
        float3 normalU = float3(dist - DistanceToObject(pos - smallfloat.xyy).x,
                           dist - DistanceToObject(pos - smallfloat.yxy).x,
                           dist - DistanceToObject(pos - smallfloat.yyx).x);
        //float3 normalU = float3(0.0);
        //for( int i=ZERO_TRICK; i<4; i++ )
        //{
        //    float3 e = 0.5773*(2.0*float3((((i+3)>>1)&1),((i>>1)&1),(i&1))-1.0);
        //    float dto = DistanceToObject(pos+0.0005*e).x;
        //    normalU += e*dto;
        //}
        normal = normalize(normalU);

        // Giant hack for border between water and snow to look nice.
        if (abs(distAndMat.z) < smallVal*0.25) normal = float3(0.0, 0.2, 0.0);

        // calculate 2 ambient occlusion values. One for global stuff and one
        // for local stuff
        float ambientS = 1.0;
        float3 distAndMatA = DistanceToObject(pos + normal * 0.4);
        ambientS *= saturate(distAndMatA.x*2.5);
        distAndMatA = DistanceToObject(pos + normal * 0.8);
        ambientS *= saturate(distAndMatA.x*1.25);
        float ambient = ambientS;
        ambient *= saturate(DistanceToObject(pos + normal * 1.6).x*1.25*0.5);
        ambient *= saturate(DistanceToObject(pos + normal * 3.2).x*1.25*0.25);
        //ambient = max(0.05, pow(ambient, 0.5));   // tone down ambient with a pow and min clamp it.
        ambient = max(0.15, ambient);
        ambient = saturate(ambient);

        // Trace a ray toward the sun for sun shadows
        float sunShadow = 1.0;
        float iter = 0.2;
        for (int i = ZERO_TRICK; i < 10; i++)
        {
            float tempDist = DistanceToObject(pos + sunDir * iter).x;
            sunShadow *= saturate(tempDist*10.0);
            if (tempDist <= 0.0) break;
            //iter *= 1.5;  // constant is more reliable than distance-based
            iter += max(0.1, tempDist)*1.2;
        }
        sunShadow = saturate(sunShadow);

        // Trace a ray through the solid for sub-surface scattering
        float scattering = 1.0;
        iter = 0.05;
        for (int i = ZERO_TRICK; i < 8; i++)
        {
            float tempDist = -DistanceToObject(pos - normal * iter).x;
            scattering *= saturate(tempDist*10.0);
            if (tempDist <= 0.0) break;
            //iter *= 1.5;  // constant is more reliable than distance-based
            iter += max(0.001, tempDist);//*0.6;
        }
        scattering = saturate(scattering);
        //scattering = (1.0-sunShadow)*(1.0-length(sunDir * iter));
        scattering = saturate(1.0-iter);

        // calculate the reflection floattor for highlights
        float3 ref = reflect(rayfloat, normal);

        // ------ Calculate tex2D color  ------
        float3 texColor = float3(0.0,0,0);
        // Underwater green glow
        float fade = 1.0-saturate(abs(distAndMat.z)*3.5);
        float3 greenFade = lerp(float3(0.1, 0.995, 0.65)*0.95, float3(0.75, 1.0, 1.0), fade*fade);
        texColor += greenFade * fade;
        texColor *= 0.75;
        // iceberg
        if (distAndMat.y == 0.0) {
            texColor = float3(1.0,1,1);
        }
        texColor = saturate(texColor);

        // ------ Calculate lighting color ------
        // Start with sun color, standard lighting equation, and shadow
        float3 lightColor = float3(14.0,14,14)*sunCol * saturate(dot(sunDir, normal)) * (sunShadow*0.7+0.3);
        // weighted average the near ambient occlusion with the far for just the right look

        // apply the light to the tex2D.
        finalColor = texColor * lightColor;
        float3 underwaterGlow = float3(0.002, 0.6, 0.51);
        // water
        if (distAndMat.y == 1.0) {
            finalColor += underwaterGlow*0.25 * length(normalU)*80.0;
            finalColor += float3(0.02, 0.5, 0.71)*0.35 * saturate(1.0-ambient);
            finalColor += skyCol*0.02 * ambientS;
        }
        // iceberg
        if (distAndMat.y == 0.0) {
            float fade = saturate(1.0-pos.y);
            // Add sky color
            finalColor += (skyCol*0.6 + horizonCol*0.4)*1.5 * saturate(normal.y *0.5+0.5);
            float3 rotPos = RotateZ(pos, sin(localTime)*sway);
            float noiseScatter = 0.0;
            float tempScale = 1.0;
            for (int i = 0; i < 3; i++) {
                noiseScatter += noiseValue(rotPos*32.0/tempScale)*0.25*tempScale;
                tempScale *= 2.0;
            }
//            noiseValue(rotPos*32.0)*0.25 +
  //              noiseValue(rotPos*16.0)*0.5 +
    //            noiseValue(rotPos*8.0)*1.0;
            finalColor += groundCol * 0.5 * max(-normal.y*0.5+0.5, 0.0) * (noiseScatter*0.3+0.6);
            finalColor += underwaterGlow * 0.35 * saturate(0.5-saturate(abs(pos.y*3.0)));
            finalColor += float3(0.01, 0.55, 0.7) * saturate(scattering-sunShadow*0.3)*0.25;
            finalColor = lerp((underwaterGlow + float3(0.5, 0.9, 0.8))*0.5, finalColor, saturate(distAndMat.z*64.0)*0.75+0.25);
            finalColor *= 0.7;
        }

        // visualize length of gradient of distance field to check distance field correctness
        //finalColor = float3(0.5) * (length(normalU) / smallfloat.x);
    }
    else
    {
        // Our ray trace hit nothing, so draw background.
        finalColor = GetEnvMap(rayfloat, sunDir);
        distAndMat.y = -1.0;
    }
    return finalColor;
}

// Input is UV coordinate of pixel to render.
// Output is RGB color.
//float3 RayTrace(in float2 fragCoord )
float3 RayTrace(in float3 ro, in float3 rd )
{
    fade = 1.0;

    float3 camPos, camUp, camLookat;
    // ------------------- Set up the camera rays for ray marching --------------------
    // Map uv to [-1.0..1.0]
    float2 uv = float2(0,0);// = //fragCoord.xy *2.0 - 1.0;   // fragCoord.xy/iResolution.xy * 2.0 - 1.0;
    uv /= 2.0;  // zoom in

#ifdef MANUAL_CAMERA
    // Camera up floattor.
    camUp=float3(0,1,0);

    // Camera lookat.
    camLookat=rd; //float3(0,0,0);

    // debugging camera
//    float mx=-iMouse.x/iResolution.x*PI*2.0;
//    float my=iMouse.y/iResolution.y*3.14*0.95 + PI/2.0;
    camPos = ro; //float3(cos(my)*cos(mx),sin(my),cos(my)*sin(mx))*5.0;
#else
    // Do the camera fly-by animation and different scenes.
    // Time variables for start and end of each scene
    const float t0 = 0.0;
    const float t1 = 12.0;
    const float t2 = 20.0;
    const float t3 = 38.0;
    // Repeat the animation after time t3
    localTime = frac(localTime / t3) * t3;
    if (localTime < t1)
    {
        float time = localTime - t0;
        float alpha = time / (t1 - t0);
        fade = saturate(time);
        fade *= saturate(t1 - localTime);
        camPos = float3(0.0, 0.4, -8.0);
        camPos.x -= smoothstep(0.0, 1.0, alpha) * 2.0;
        camPos.y += smoothstep(0.0, 1.0, alpha) * 2.0;
        camPos.z += smoothstep(0.0, 1.0, alpha) * 4.0;
        camUp=float3(0,1,0);
        camLookat=float3(0,-0.5,0.5);
        camLookat.y -= smoothstep(0.0, 1.0, alpha) * 0.5;
    } else if (localTime < t2)
    {
        float time = localTime - t1;
        float alpha = time / (t2 - t1);
        fade = saturate(time);
        fade *= saturate(t2 - localTime);
        camPos = float3(2.0, 4.3, -0.5);
        camPos.y -= alpha * 3.5;
        camPos.x = sin(alpha*1.0) * 9.2;
        camPos.z = cos(alpha*1.0) * 6.2;
        camUp=normalize(float3(0,1,-0.005 + alpha * 0.005));
        camLookat=float3(0,-1.5,0.0);
        camLookat.y += smoothstep(0.0, 1.0, alpha) * 1.5;
    } else if (localTime < t3)
    {
        float time = localTime - t2;
        float alpha = time / (t3 - t2);
        fade = saturate(time);
        fade *= saturate(t3 - localTime);
        camPos = float3(-9.0, 1.3, -10.0);
        //camPos.y -= alpha * 8.0;
        camPos.x += alpha * 14.0;
        camPos.z += alpha * 7.0;
        camUp=normalize(float3(0,1,0.0));
        camLookat=float3(0.0,0.0,0.0);
    }
#endif

    // Camera setup for ray tracing / marching
    float3 camfloat=normalize(camLookat - camPos);
    float3 sideNorm=normalize(cross(camUp, camfloat));
    float3 upNorm=cross(camfloat, sideNorm);
    float3 worldFacing=(camPos + camfloat);
    float3 worldPix = worldFacing + uv.x * sideNorm + uv.y * upNorm;
    float3 rayfloat = normalize(worldPix - camPos);

    float3 finalColor = float3(0.0,0,0);

    float3 normal;
    float3 distAndMat;
    float t;
    finalColor = TraceOneRay(camPos, rayfloat, normal, distAndMat, t);
    float origDelta = distAndMat.z;
    if (distAndMat.y == 1.0) {
        float3 ref = normalize(reflect(rayfloat, normal));
        ref.y = abs(ref.y);
        float3 newStartPos = (camPos + rayfloat * t) + normal * 0.02; // nudge away.
        float fresnel = saturate(1.0 - dot(-rayfloat, normal));
        fresnel = fresnel * fresnel * fresnel * fresnel * fresnel * fresnel;
        fresnel = lerp(0.05, 0.9, fresnel);
        float3 refColor = TraceOneRay(newStartPos, ref, normal, distAndMat, t);
        finalColor += refColor * fresnel;
    }

    // vignette?
    finalColor *= float3(1.0,1,1) * saturate(1.0 - length(uv/2.5));
    finalColor *= exposure;

    // output the final color without gamma correction - will do gamma later.
    return float3(clamp(finalColor, 0.0, 1.0));//*saturate(fade));
}

#ifdef NON_REALTIME_HQ_RENDER
// This function breaks the image down into blocks and scans
// through them, rendering 1 block at a time. It's for non-
// realtime things that take a long time to render.

// This is the frame rate to render at. Too fast and you will
// miss some blocks.
const float blockRate = 20.0;
void BlockRender(in float2 fragCoord)
{
    // blockSize is how much it will try to render in 1 frame.
    // adjust this smaller for more complex scenes, bigger for
    // faster render times.
    const float blockSize = 64.0;
    // Make the block repeatedly scan across the image based on time.
    float frame = floor(_Time.y * blockRate);
    float2 blockRes = floor(iResolution.xy / blockSize) + float2(1.0);
    // ugly bug with mod.
    //float blockX = mod(frame, blockRes.x);
    float blockX = frac(frame / blockRes.x) * blockRes.x;
    //float blockY = mod(floor(frame / blockRes.x), blockRes.y);
    float blockY = frac(floor(frame / blockRes.x) / blockRes.y) * blockRes.y;
    // Don't draw anything outside the current block.
    if ((fragCoord.x - blockX * blockSize >= blockSize) ||
        (fragCoord.x - (blockX - 1.0) * blockSize < blockSize) ||
        (fragCoord.y - blockY * blockSize >= blockSize) ||
        (fragCoord.y - (blockY - 1.0) * blockSize < blockSize))
    {
        discard;
    }
}
#endif


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

#ifdef NON_REALTIME_HQ_RENDER
    // Optionally render a non-realtime scene with high quality
    BlockRender(fragCoord);
#endif

    // Do a multi-pass render
    float3 finalColor = float3(0.0,0,0);
#ifdef NON_REALTIME_HQ_RENDER
    for (float i = 0.0; i < antialiasingSamples; i++)
    {
        const float motionBlurLengthInSeconds = 1.0 / 60.0;
        // Set this to the time in seconds of the frame to render.
        localTime = frameToRenderHQ;
        // This line will motion-blur the renders
        localTime += Hash11(v21(fragCoord + seed)) * motionBlurLengthInSeconds;
        // Jitter the pixel position so we get antialiasing when we do multiple passes.
        float2 jittered = fragCoord.xy + float2(
            Hash21(fragCoord + seed),
            Hash21(fragCoord*7.234567 + seed)
            );
        // don't antialias if only 1 sample.
        if (antialiasingSamples == 1.0) jittered = fragCoord;
        // Accumulate one pass of raytracing into our pixel value
//        finalColor += RayTrace(jittered);
        // Change the random seed for each pass.
        seed *= 1.01234567;
    }
    // Average all accumulated pixel intensities
    finalColor /= antialiasingSamples;
#else
    // Regular real-time rendering
    localTime = _Time.y;
//    finalColor = RayTrace(fragCoord);
    finalColor = RayTrace(ro,rd);
#endif

    fragColor = float4(sqrt(clamp(finalColor, 0.0, 1.0)),1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}


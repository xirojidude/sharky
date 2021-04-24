
Shader "Skybox/FluxCore"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Vector) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Vector) = (0, 15, -.25 ,0) 
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

            uniform sampler2D _MainTex; 
            float4 _SunDir,_XYZPos;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 worldPos : TEXCOORD1;
                float4 vertex : SV_POSITION;   //pos
                float4 screenPos : TEXCOORD2;
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                o.worldPos = mul(unity_ObjectToWorld, v2.vertex);
                o.screenPos  = ComputeScreenPos(o.vertex);
                //o.screenPos.z = -(mul(UNITY_MATRIX_MV, vertex).z * _ProjectionParams.w);     // old calculation, as I used the depth buffer comparision for min max ray march. 

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
const float frameToRenderHQ = 20.0; // Time in seconds of frame to render
const float antialiasingSamples = 16.0; // 16x antialiasing - too much might make the shader compiler angry.

//#define MANUAL_CAMERA


#define ZERO_TRICK max(0, -iFrame)
// --------------------------------------------------------
// These variables are for the non-realtime block renderer.
float localTime = 0.0;
float seed = 1.0;

// Animation variables
float animStructure = 1.0;
float fade = 1.0;

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
float Hash1d(float u)
{
    return frac(sin(u)*143.9); // scale this down to kill the jitters
}
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
float noise(float3 uv)
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
  float cos0 = cos(rad);
  float sin0 = sin(rad);
  return float3(v.x, cos0 * v.y + sin0 * v.z, -sin0 * v.y + cos0 * v.z);
}
float3 RotateY(float3 v, float rad)
{
  float cos0 = cos(rad);
  float sin0 = sin(rad);
  return float3(cos0 * v.x - sin0 * v.z, v.y, sin0 * v.x + cos0 * v.z);
}
float3 RotateZ(float3 v, float rad)
{
  float cos0 = cos(rad);
  float sin0 = sin(rad);
  return float3(cos0 * v.x + sin0 * v.y, -sin0 * v.x + cos0 * v.y, v.z);
}

// This spiral noise works by successively adding and rotating sin waves while increasing frequency.
// It should work the same on all computers since it's not based on a hash function like some other noises.
// It can be much faster than other noise functions if you're ok with some repetition.
const float nudge = 0.71;   // size of perpendicular Vector
float normalizer = 1.0 / 1.226417547;  //sqrt(1.0 + nudge*nudge);   // pythagorean theorem on that perpendicular to maintain scale
// Total hack of the spiral noise function to get a rust look
float RustNoise3D(float3 p)
{
    float n = 0.0;
    float iter = 1.0;
    float pn = noise(p*0.125);
    pn += noise(p*0.25)*0.5;
    pn += noise(p*0.5)*0.25;
    pn += noise(p*1.0)*0.125;
    for (int i = 0; i < 7; i++) //ZERO_TRICK
    {
        //n += (sin(p.y*iter) + cos(p.x*iter)) / iter;
        float wave = saturate(cos(p.y*0.25 + pn) - 0.998);
        wave *= noise(p * 0.125)*1016.0;
        n += wave;
        p.xy += float2(p.y, -p.x) * nudge;
        p.xy *= normalizer;
        p.xz += float2(p.z, -p.x) * nudge;
        p.xz *= normalizer;
        iter *= 1.4733;
    }
    return n;
}

// ---- functions to remap / warp space ----
float repsDouble(float a)
{
    return abs(a * 2.0 - 1.0);
}
float2 repsDouble(float2 a)
{
    return abs(a * 2.0 - 1.0);
}

float2 mapSpiralMirror(float2 uv)
{
    float len = length(uv);
    float at = atan2( uv.y,uv.x);
    at = at / PI;
    float dist = (frac(log(len)+at*0.5)-0.5) * 2.0;
    at = repsDouble(at);
    at = repsDouble(at);
    return float2(abs(dist), abs(at));
}

float2 mapSpiral(float2 uv)
{
    float len = length(uv);
    float at = atan2( uv.y,uv.x);
    at = at / PI;
    float dist = (frac(log(len)+at*0.5)-0.5) * 2.0;
    //dist += sin(at*32.0)*0.05;
    // at is [-1..1]
    // dist is [-1..1]
    at = repsDouble(at);
    at = repsDouble(at);
    return float2(dist, at);
}

float2 mapCircleInvert(float2 uv)
{
    float len = length(uv);
    float at = atan2( uv.y,uv.x);
    //at = at / PI;
    //return uv;
    len = 1.0 / len;
    return float2(sin(at)*len, cos(at)*len);
}

float3 mapSphereInvert(float3 uv)
{
    float len = length(uv);
    float3 dir = normalize(uv);
    len = 1.0 / len;
    return dir * len;
}

// ---- shapes defined by distance fields ----
// See this site for a reference to more distance functions...
// http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float length8(float2 v)
{
    return pow(pow(abs(v.x),8.0) + pow(abs(v.y), 8.0), 1.0/8.0);
}

// box distance field
float sdBox(float3 p, float3 radius)
{
  float3 dist = abs(p) - radius;
  return min(max(dist.x, max(dist.y, dist.z)), 0.0) + length(max(dist, 0.0));
}

// Makes a warped torus that rotates around
float sdTorusWobble( float3 p, float2 t, float offset)
{
    float a = atan2( p.z,p.x);
    float subs = 2.0;
    a = sin(a*subs+localTime*4.0+offset*3.234567);
    float2 q = float2(length(p.xz)-t.x-a*0.1,p.y);
    return length8(q)-t.y;
}

// simple cylinder distance field
float cyl(float2 p, float r)
{
    return length(p) - r;
}

float glow = 0.0, glow2 = 0.0, glow3 = 0.0;
float pulse;
// This is the big money function that makes the crazy fracally shape
// The input is a position in space.
// The output is the distance to the nearest surface.
float DistanceToObject(float3 p)
{
    float3 orig = p;
    // Magically remap space to be in a spiral
    p.yz = mapSpiralMirror(p.yz);
    // lerp between spiral space and unwarped space. This changes the scene
    // from the tunnel to the spiral.
    p = lerp(orig, p, animStructure);
//    p = lerp(p, orig, cos(localTime)*0.5+0.5);

    // Cut out stuff outside of outer radius
    const float outerRad = 3.5;
    float lenXY = length(p.xy);
    float final = lenXY - outerRad;
    // Carve out inner radius
    final = max(final, -(lenXY - (outerRad-0.65)));

    // Slice the object in a 3d grid
    float slice = 0.04;
    float3 grid = -abs(frac(p)-0.5) + slice;
    //final = max(final, grid.x);
    //final = max(final, grid.y);
    final = max(final, grid.z);

    // Carve out cylinders from the object on all 3 axis, scaled 3 times
    // This gives it the fracal look.
    float3 rep = frac(p)-0.5;
    float scale = 1.0;
    float mult = 0.32;
    for (int i = 0; i < 3; i++) // ZERO_TRICK
    {
        float uglyDivider = max(1.0, float(i)); // wtf is this? My math sucks :(
        // carve out 3 cylinders
        float dist = cyl(rep.xz/scale, mult/scale)/uglyDivider;
        final = max(final, -dist);
        dist = cyl(rep.xy/scale, mult/scale)/uglyDivider;
        final = max(final, -dist);
        dist = cyl(rep.yz/scale, mult/scale)/uglyDivider;
        final = max(final, -dist);
        // Scale and repeat.
        scale *= 1.14+1.0;// + sin(localTime)*0.995;
        rep = frac(rep*scale) - 0.5;
    }

    // Make radial struts that poke into the center of the spiral
    float3 sp = p;
    sp.x = abs(sp.x)-5.4;
    sp.z = frac(sp.z) - 0.5;
    // Bad distance field on these makes them sometimes disappear. Math. :(
    float struts = sdBox(sp+float3(2.95, 0.1-sin(sp.x*2.0)*1.1, 0.0), float3(1.5, 0.05, 0.02))*0.5;
    //glow3 += (0.00005)/max(0.01, struts);
    final = min(final, struts);

    // Make spiral glows that rotate and pulse energy to the center
    rep.yz = (frac(p.yz)-0.5);
    rep.x = p.x;
    scale = 1.14+1.0;
    float jolt = max(0.0, sin(length(orig.yz) + localTime*20.0))*0.94;
    jolt *= saturate(0.3-pulse);
    float spiral = sdBox(RotateX(rep+float3(-0.05,0.0,0.0), pulse), float3(0.01+jolt,1.06, mult*0.01)/scale );
    glow3 += (0.0018)/max(0.0025,spiral);
    final = min(final, spiral + (1.0-animStructure) * 100.0);

    // Make a warped torus that rotates around and glows orange
    float3 rp = p.xzy;
    rp.x = -abs(rp.x);
    rp.y = frac(rp.y) - 0.5;
    float torus = sdTorusWobble(rp + float3(3.0, 0.0, 0.0), float2(0.2, 0.0003), p.z);
    glow2 += 0.0015 / max(0.03, torus);
    final = min(final, torus);

    // Make the glowing tower in the center.
    // This also gives a bit of a glow to everything.
    glow += (0.02+abs(sin(orig.x-localTime*3.0)*0.15)*jolt )/length(orig.yz);

    return final;
}

// Input is UV coordinate of pixel to render.
// Output is RGB color.
float3 RayTrace(in float2 fragCoord )
{
    glow = 0.0;
    glow2 = 0.0;
    glow3 = 0.0;
    // -------------------------------- animate ---------------------------------------
    // Default to spiral shape
    animStructure = 1.0;

    // Make a cycling, clamped sin wave to animate the glow-spiral rotation.
    float slt = sin(localTime);
    float stepLike = pow(abs(slt), 0.75)*sign(slt);
    stepLike = max(-1.0, min(1.0, stepLike*1.5));
    pulse = stepLike*PI/4.0 + PI/4.0;

    float3 camPos, camUp, camLookat;
    // ------------------- Set up the camera rays for ray marching --------------------
    // Map uv to [-1.0..1.0]
    float2 uv =  fragCoord.xy*2-1;  // /iResolution.xy * 2.0 - 1.0;

#ifdef MANUAL_CAMERA
    // Camera up Vector.
    camUp=float3(0,1,0);

    // Camera lookat.
    camLookat=float3(0,0.0,0);

    // debugging camera
    float mx=iMouse.x/iResolution.x*PI*2.0;// + localTime * 0.166;
    float my=-iMouse.y/iResolution.y*10.0;// + sin(localTime * 0.3)*0.8+0.1;//*PI/2.01;
    camPos = float3(cos(my)*cos(mx),sin(my),cos(my)*sin(mx))*8.35;
#else
    // Do the camera fly-by animation and different scenes.
    // Time variables for start and end of each scene
    const float t0 = 0.0;
    const float t1 = 9.0;
    const float t2 = 16.0;
    const float t3 = 24.0;
    const float t4 = 40.0;
    const float t5 = 48.0;
    const float t6 = 70.0;
    // Repeat the animation after time t6
    localTime = frac(localTime / t6) * t6;
    /*const float t0 = 0.0;
    const float t1 = 0.0;
    const float t2 = 0.0;
    const float t3 = 0.0;
    const float t4 = 0.0;
    const float t5 = 0.0;
    const float t6 = 18.0;*/
    if (localTime < t1)
    {
        animStructure = 0.0;
        float time = localTime - t0;
        float alpha = time / (t1 - t0);
        fade = saturate(time);
        fade *= saturate(t1 - localTime);
        camPos = float3(56.0, -2.5, 1.5);
        camPos.x -= alpha * 6.8;
        camUp=float3(0,1,0);
        camLookat=float3(50,0.0,0);
    } else if (localTime < t2)
    {
        animStructure = 0.0;
        float time = localTime - t1;
        float alpha = time / (t2 - t1);
        fade = saturate(time);
        fade *= saturate(t2 - localTime);
        camPos = float3(12.0, 3.3, -0.5);
        camPos.x -= smoothstep(0.0, 1.0, alpha) * 4.8;
        camUp=float3(0,1,0);
        camLookat=float3(0,5.5,-0.5);
    } else if (localTime < t3)
    {
        animStructure = 1.0;
        float time = localTime - t2;
        float alpha = time / (t3 - t2);
        fade = saturate(time);
        fade *= saturate(t3 - localTime);
        camPos = float3(12.0, 6.3, -0.5);
        camPos.y -= alpha * 1.8;
        camPos.x = cos(alpha*1.0) * 6.3;
        camPos.z = sin(alpha*1.0) * 6.3;
        camUp=normalize(float3(0,1,-0.3 - alpha * 0.5));
        camLookat=float3(0,0.0,-0.5);
    } else if (localTime < t4)
    {
        animStructure = 1.0;
        float time = localTime - t3;
        float alpha = time / (t4 - t3);
        fade = saturate(time);
        fade *= saturate(t4 - localTime);
        camPos = float3(12.0, 3.0, -2.6);
        camPos.y -= alpha * 1.8;
        camPos.x = cos(alpha*1.0) * 6.5-alpha*0.25;
        camPos.z += sin(alpha*1.0) * 6.5-alpha*0.25;
        camUp=normalize(float3(0,1,0.0));
        camLookat=float3(0,0.0,-0.0);
    } else if (localTime < t5)
    {
        animStructure = 1.0;
        float time = localTime - t4;
        float alpha = time / (t5 - t4);
        fade = saturate(time);
        fade *= saturate(t5 - localTime);
        camPos = float3(0.0, -7.0, -0.9);
        camPos.y -= alpha * 1.8;
        camPos.x = cos(alpha*1.0) * 1.5-alpha*1.5;
        camPos.z += sin(alpha*1.0) * 1.5-alpha*1.5;
        camUp=normalize(float3(0,1,0.0));
        camLookat=float3(0,-3.0,-0.0);
    } else if (localTime < t6)
    {
        float time = localTime - t5;
        float alpha = time / (t6 - t5);
        float smoothv = smoothstep(0.0, 1.0, saturate(alpha*1.8-0.1));
        animStructure = 1.0-smoothv;
        fade = saturate(time);
        fade *= saturate(t6 - localTime);
        camPos = float3(10.0, -0.95+smoothv*1.0, 0.0);
        camPos.x -= alpha * 6.8;
        camUp=normalize(float3(0,1.0-smoothv,0.0+smoothv));
        camLookat=float3(0,-0.0,-0.0);
    }
#endif

    // Camera setup.
    float3 camfloat=normalize(camLookat - camPos);
    float3 sideNorm=normalize(cross(camUp, camfloat));
    float3 upNorm=cross(camfloat, sideNorm);
    float3 worldFacing=(camPos + camfloat);
    float3 worldPix = worldFacing + uv.x * sideNorm + uv.y *upNorm;   // * (iResolution.x/iResolution.y) + uv.y * upNorm;
    float3 rayfloat = normalize(worldPix - camPos);

    // ----------------------------- Ray march the scene ------------------------------
    float dist = 1.0;
    float t = 0.1 + Hash2d(uv)*0.1; // random dither-fade things close to the camera
    const float maxDepth = 45.0; // farthest distance rays will travel
    float3 pos = float3(0,0,0);
    const float smallVal = 0.000625;
    // ray marching time
    for (int i = 0; i < 210; i++)  // ZERO_TRICK  // This is the count of the max times the ray actually marches.
    {
        // Step along the ray. Switch x, y, and z because I messed up the orientation.
        pos = (camPos + rayfloat * t).yzx;
        // This is _the_ function that defines the "distance field".
        // It's really what makes the scene geometry. The idea is that the
        // distance field returns the distance to the closest object, and then
        // we know we are safe to "march" along the ray by that much distance
        // without hitting anything. We repeat this until we get really close
        // and then break because we have effectively hit the object.
        dist = DistanceToObject(pos);
        // This makes the ray trace more precisely in the center so it will not miss the
        // vertical glowy beam.
        dist = min(dist, length(pos.yz));

        t += dist;
        // If we are very close to the object, let's call it a hit and exit this loop.
        if ((t > maxDepth) || (abs(dist) < smallVal)) break;
    }

    // --------------------------------------------------------------------------------
    // Now that we have done our ray marching, let's put some color on this geometry.
    float glowSave = glow;
    float glow2Save = glow2;
    float glow3Save = glow3;

    float3 sunDir = normalize(float3(0.93, 1.0, -1.5));
    float3 finalColor = float3(0.0,0,0);

    // If a ray actually hit the object, let's light it.
    if (t <= maxDepth)
    {
        // calculate the normal from the distance field. The distance field is a volume, so if you
        // sample the current point and neighboring points, you can use the difference to get
        // the normal.
        float3 smallfloat = float3(smallVal, 0, 0);
        float3 normalU = float3(dist - DistanceToObject(pos - smallfloat.xyy),
                           dist - DistanceToObject(pos - smallfloat.yxy),
                           dist - DistanceToObject(pos - smallfloat.yyx));
        float3 normal = normalize(normalU);

        // calculate 2 ambient occlusion values. One for global stuff and one
        // for local stuff
        float ambientS = 1.0;
        ambientS *= saturate(DistanceToObject(pos + normal * 0.05)*20.0);
        ambientS *= saturate(DistanceToObject(pos + normal * 0.1)*10.0);
        ambientS *= saturate(DistanceToObject(pos + normal * 0.2)*5.0);
        ambientS *= saturate(DistanceToObject(pos + normal * 0.4)*2.5);
        ambientS *= saturate(DistanceToObject(pos + normal * 0.8)*1.25);
        float ambient = ambientS * saturate(DistanceToObject(pos + normal * 1.6)*1.25*0.5);
        //ambient *= saturate(DistanceToObject(pos + normal * 3.2)*1.25*0.25);
        //ambient *= saturate(DistanceToObject(pos + normal * 6.4)*1.25*0.125);
        //ambient = max(0.05, pow(ambient, 0.3));   // tone down ambient with a pow and min clamp it.
        ambient = saturate(ambient);

        // calculate the reflection Vector for highlights
        //float3 ref = reflect(rayfloat, normal);

        // Trace a ray toward the sun for sun shadows
        float sunShadow = 1.0;
        float iter = 0.01;
        float3 nudgePos = pos + normal*0.002; // don't start tracing too close or inside the object
        for (int i = 0; i < 30; i++)    // ZERO_TRICK
        {
            float tempDist = DistanceToObject(nudgePos + sunDir * iter);
            sunShadow *= saturate(tempDist*150.0);  // Shadow hardness
            if (tempDist <= 0.0) break;
            //iter *= 1.5;  // constant is more reliable than distance-based
            iter += max(0.01, tempDist)*1.0;
            if (iter > 4.2) break;
        }
        sunShadow = saturate(sunShadow);

        // make a few frequencies of noise to give it some texture
        float n =0.0;
        n += noise(pos*32.0);
        n += noise(pos*64.0);
        n += noise(pos*128.0);
        n += noise(pos*256.0);
        n += noise(pos*512.0);
        n *= 0.8;
        normal = normalize(normal + (n-2.0)*0.1);

        // ------ Calculate texture color  ------
        float3 texColor = float3(0.95, 1.0, 1.0);
        float3 rust = float3(0.65, 0.25, 0.1) - noise(pos*128.0);
        // Call the function that makes rust stripes on the texture
        float rn = saturate(RustNoise3D(pos*8.0))-0.2;
        texColor *= smoothstep(texColor, rust, float3(rn,rn,rn));

        // apply noise
        texColor *= float3(1.0,1,1)*n*0.05;
        texColor *= 0.7;
        texColor = saturate(texColor);

        // ------ Calculate lighting color ------
        // Start with sun color, standard lighting equation, and shadow
        float3 lightColor = float3(3.6,3.6,3.6) * saturate(dot(sunDir, normal)) * sunShadow;
        // weighted average the near ambient occlusion with the far for just the right look
        float ambientAvg = (ambient*3.0 + ambientS) * 0.25;
        // a red and blue light coming from different directions
        lightColor += (float3(1.0, 0.2, 0.4) * saturate(-normal.z *0.5+0.5))*pow(ambientAvg, 0.35);
        lightColor += (float3(0.1, 0.5, 0.99) * saturate(normal.y *0.5+0.5))*pow(ambientAvg, 0.35);
        // blue glow light coming from the glow in the middle
        lightColor += float3(0.3, 0.5, 0.9) * saturate(dot(-pos, normal))*pow(ambientS, 0.3);
        lightColor *= 4.0;

        // finally, apply the light to the texture.
        finalColor = texColor * lightColor;
        // sun reflection to make it look metal
        //finalColor += float3(1.0)*pow(n,4.0)* GetSunColorSmall(ref, sunDir) * sunShadow;// * ambientS;
        // visualize length of gradient of distance field to check distance field correctness
        //finalColor = float3(0.5) * (length(normalU) / smallfloat.x);
    }
    else
    {
        // Our ray trace hit nothing, so draw sky.
    }
    // add the ray marching glows
    float center = length(pos.yz);
    finalColor += float3(0.3, 0.5, 0.9) * glowSave*1.2;
    finalColor += float3(0.9, 0.5, 0.3) * glow2*1.2;
    finalColor += float3(0.25, 0.29, 0.93) * glow3Save*2.0;

    // vignette?
    finalColor *= float3(1.0,1,1) * saturate(1.0 - length(uv/2.5));
    finalColor *= 1.0;// 1.3;

    // output the final color without gamma correction - will do gamma later.
    return float3(clamp(finalColor, 0.0, 1.0)*saturate(fade+0.25));
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
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

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
        finalColor += RayTrace(jittered);
        // Change the random seed for each pass.
        seed *= 1.01234567;
    }
    // Average all accumulated pixel intensities
    finalColor /= antialiasingSamples;
#else
    // Regular real-time rendering
    localTime = _Time.y;
    finalColor = RayTrace(fragCoord);
#endif

    fragColor = float4(sqrt(clamp(finalColor, 0.0, 1.0)),1.0);

                return fragColor;
            }

            ENDCG
        }
    }
}



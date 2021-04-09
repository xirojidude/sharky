
Shader "Skybox/DesertSand"
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

// https://www.shadertoy.com/view/ld3BzM

/*
    Desert Sand
    -----------

    Sand, and more sand -- Monotony, at it's finest. :) I've visited and flown over many sandy 
    regions in my time, and I can say unequivocably that this particular scene doesn't remotely  
    resemble the real thing. :D 

    Having said that, there's something that I really like about minimal artificial dune scenes.  
    They're somewhat of a visual programmer cliche -- I think that's partly due to the fact that 
    they have a decent "aesthetic appeal to algorithmic complexity" ratio.
    
    For the sand dunes, rather than deal with dune physics, I used a pretty standard smoothstep 
    noise layer formula. I wouldn't say that dune physics is particularly difficult, but I'd 
    rather leave that to people like Dr2. :) Besides, with this example, I wanted to save some
    cycles and focus on the sand texture generation.

    There are so many different ways to create wavy sand patterns. Some are expensive -- using
    things like gradient erosion -- and some are cheap. Not suprisingly, the expensive methods 
    tend to look better. I bump mapped the sand layer to give myself a few extra cycles to play 
    with, but I still had to keep things relatively simple.

    The pattern you see is a lerpture of a simple trick I've seen around and some of my own
    adjustments. Without going into detail, the idea is to create a layer of repeat rounded 
    gradient lines, and another rotated at a slight angle, then perturb them slightly and lerp 
    together using an underlying noise layer. It's simple, but reasonably effective. Anyway, 
    I've explained in more detail in the "sand" function below.

    By the way, with some tweaking, grouping, rearranging, etc, I'm pretty sure I could make this 
    run faster. However, it'd be at the expense of the readability of the sand texture function,
    so for now, I'm going to leave it alone. Anyway, I'm going to produce a few surface realated
    examples later.
    

    Related examples:

    // It won Breakpoint way back in 2009. Ten people on Pouet gave it the thumbs down -- I hope
    // they put their work up on Shadertoy, because I'd imagine it'd be insanely good. :D
    Elevated - IQ
    https://www.shadertoy.com/view/MdX3Rr


    // One of my favorite simple coloring jobs.
    Skin Peeler - Dave Hoskins
    https://www.shadertoy.com/view/XtfSWX
    Based on one of my all time favorites:
    Xyptonjtroz - Nimitz
    https://www.shadertoy.com/view/4ts3z2

*/

// The far plane. I'd like this to be larger, but the extra iterations required to render the 
// additional scenery starts to slow things down on my slower machine.
#define FAR 80.


// Fabrice's concise, 2D rotation formula.
//float2x2 rot2(float th){ float2 a = sin(float2(1.5707963, 0) + th); return float2x2(a, -a.y, a.x); }
// Standard 2D rotation formula - Nimitz says it's faster, so that's good enough for me. :)
float2x2 rot2(in float a){ float c = cos(a), s = sin(a); return float2x2(c, s, -s, c); }


// 3x1 hash function.
float hash( float3 p ){ return frac(sin(dot(p, float3(21.71, 157.97, 113.43)))*45758.5453); }


// IQ's smooth minium function. 
float smin(float a, float b , float s){
    
    float h = clamp( 0.5 + 0.5*(b-a)/s, 0. , 1.);
    return lerp(b, a, h) - h*(1.0-h)*s;
}

// Smooth maximum, based on IQ's smooth minimum.
float smax(float a, float b, float s){
    
    float h = clamp( 0.5 + 0.5*(a-b)/s, 0., 1.);
    return lerp(b, a, h) + h*(1.0-h)*s;
}


// Dave's hash function. More reliable with large values, but will still eventually break down.
//
// Hash without Sine
// Creative Commons Attribution-ShareAlike 4.0 International Public License
// Created by David Hoskins.
// float2 to float2.
float2 hash22(float2 p){

    float3 p3 = frac(float3(p.xyx) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 19.19);
    p = frac((p3.xx + p3.yz)*p3.zy)*2. - 1.;
    return p;
    
    
    // Note the "mod" call. Slower, but ensures accuracy with large time values.
    //float2x2  m = r2(((_Time.y)% 6.2831853)); 
    //p.xy = m * p.xy;//rotate gradient floattor
    //return p;
    

}


/*
#define RIGID
// Standard 2x2 hash algorithm.
float2 hash22(float2 p) {
    
    // Faster, but probaly doesn't disperse things as nicely as other methods.
    float n = sin(dot(p, float2(113, 1)));
    p = frac(float2(2097152, 262144)*n)*2. - 1.;
    #ifdef RIGID
    return p;
    #else
    return cos(p*6.283 + iGlobalTime);
    //return abs(frac(p+ iGlobalTime*.25)-.5)*2. - .5; // Snooker.
    //return abs(cos(p*6.283 + iGlobalTime))*.5; // Bounce.
    #endif

}
*/

float2 nmzHash22(float2 q)
{
    uint2 p =  uint2(asint(q)); //uint2(ivec2(q));
    p = p*uint2(3266489917U, 668265263U) + p.yx;
    p = p*(p.yx^(p >> 15U));
    float2 a = float2(p^(p >> 16U));
    uint2 b = (1.0/float2(0xffffffffU,0xffffffffU));
    return a*b;
}


/*
const float2 zeroOne = float2(0.0, 1.0);
float lerpP(float f0, float f1, float a)
{
    return lerp(f0, f1, a*a*(3.0-2.0*a));
}
// noise functions
float Hash2d(float2 uv)
{
    float f = uv.x + uv.y * 37.0;
    return frac(sin(f)*104003.9);
}
float gradN2D_new(float2 uv)  //noise2d
{
    float2 fr = frac(uv.xy);
    float2 fl = floor(uv.xy);
    float h00 = Hash2d(fl);
    float h10 = Hash2d(fl + zeroOne.yx);
    float h01 = Hash2d(fl + zeroOne);
    float h11 = Hash2d(fl + zeroOne.yy);
    return lerpP(lerpP(h00, h10, fr.x), lerpP(h01, h11, fr.x), fr.y);
}
*/

// Gradient noise. Ken Perlin came up with it, or a version of it. Either way, this is
// based on IQ's implementation. It's a pretty simple process: Break space into squares, 
// attach random 2D floattors to each of the square's four vertices, then smoothly 
// interpolate the space between them.
float gradN2D(in float2 f){
    
    // Used as shorthand to write things like float3(1, 0, 1) in the short form, e.yxy. 
   const float2 e = float2(0, 1);
   
    // Set up the cubic grid.
    // Integer value - unique to each cube, and used as an ID to generate random floattors for the
    // cube vertiies. Note that vertices shared among the cubes have the save random floattors attributed
    // to them.
    float2 p = floor(f);
    f -= p; // fracional position within the cube.
    

    // Smoothing - for smooth interpolation. Use the last line see the difference.
    //float2 w = f*f*f*(f*(f*6.-15.)+10.); // Quintic smoothing. Slower and more squarish, but derivatives are smooth too.
    float2 w = f*f*(3. - 2.*f); // Cubic smoothing. 
    //float2 w = f*f*f; w = ( 7. + (w - 7. ) * f ) * w; // Super smooth, but less practical.
    //float2 w = .5 - .5*cos(f*3.14159); // Cosinusoidal smoothing.
    //float2 w = f; // No smoothing. Gives a blocky appearance.
    
    // Smoothly interpolating between the four verticies of the square. Due to the shared vertices between
    // grid squares, the result is blending of random values throughout the 2D space. By the way, the "dot" 
    // operation makes most sense visually, but isn't the only metric possible.
    float c = lerp(lerp(dot(hash22(p + e.xx), f - e.xx), dot(hash22(p + e.yx), f - e.yx), w.x),
                  lerp(dot(hash22(p + e.xy), f - e.xy), dot(hash22(p + e.yy), f - e.yy), w.x), w.y);
    
    // Taking the final result, and converting it to the zero to one range.
    return c*.5 + .5; // Range: [0, 1].
}

// Gradient noise fBm.
float fBm(in float2 p){
    
    return gradN2D(p)*.57 + gradN2D(p*2.)*.28 + gradN2D(p*4.)*.15;
    
}

/////
// Code block to produce three layers of fine dust. Not sophisticated at all.
// If you'd like to see a much more sophisticated version, refer to Nitmitz's
// Xyptonjtroz example. Incidently, I wrote this off the top of my head, but
// I did have that example in mind when writing this.
float trig3(in float3 p){
    p = cos(p*2. + (cos(p.yzx) + 1.)*1.57);// + _Time.y*1.
    return dot(p, float3(0.1666,0.1666,0.1666)) + 0.5;
}

// Basic low quality noise consisting of three layers of rotated, mutated 
// trigonometric functions. Needs work, but it's OK for this example.
float trigNoise3D(in float3 p){

    // 3D transformation matrix.
    const float3x3 m3RotTheta = float3x3(0.25, -0.866, 0.433, 0.9665, 0.25, -0.2455127, -0.058, 0.433, 0.899519 )*1.5;
  
    float res = 0.;

    float t = trig3(p*3.14159265);
    p += (t);
    p = mul(m3RotTheta,p);
    //p = (p+0.7071)*1.5;
    res += t;
    
    t = trig3(p*3.14159265); 
    p += (t)*0.7071;
    p = mul(m3RotTheta,p);
     //p = (p+0.7071)*1.5;
    res += t*0.7071;

    t = trig3(p*3.14159265);
    res += t*0.5;
     
    return res/2.2071;
}


// Cheap and nasty 2D smooth noise function with inbuilt hash function - based on IQ's 
// original. Very trimmed down. In fact, I probably went a little overboard. I think it 
// might also degrade with large time values. I'll swap it for something more robust later.
float n2D_orig(float2 p) {

    float2 i = floor(p); p -= i; 
    //p *= p*p*(p*(p*6. - 15.) + 10.);
    p *= p*(3. - p*2.);  
    
    float2x2 innerfrac = float2x2( frac(
                        sin(
                            float4(0, 1, 113, 114) + dot(i, float2(1, 113))
                        )*43758.5453
                    ));
    return dot( mul(innerfrac , float2(1. - p.y, p.y))
                , float2(1. - p.x, p.x) 
                );

}


// Cheap and nasty 2D smooth noise function, based on IQ's original. Very trimmed down. In fact,
// I probably went a little overboard. I think it might also degrade with large time values. I'll 
// swap it for something more robust later.
float n2D(float2 p) {

    float2 f = frac(p); p -= f; f *= f*(3. - f*2.);  
    
    return dot(mul(float2x2(frac(sin(float4(0, 41, 289, 330) + dot(p, float2(41, 289)))*43758.5453)),
                float2(1. - f.y, f.y)), float2(1. - f.x, f.x) );

}


// Repeat gradient lines. How you produce these depends on the effect you're after. I've used a smoothed
// triangle gradient lerped with a custom smoothed gradient to effect a little sharpness. It was produced
// by trial and error. If you're not sure what it does, just call it individually, and you'll see.
float grad(float x, float offs){
    
    // Repeat triangle wave. The tau factor and ".25" factor aren't necessary, but I wanted its frequency
    // to overlap a sine function.
    x = abs(frac(x/6.283 + offs - .25) - .5)*2.;
    
    float x2 = clamp(x*x*(-1. + 2.*x), 0., 1.); // Customed smoothed, peaky triangle wave.
    //x *= x*x*(x*(x*6. - 15.) + 10.); // Extra smooth.
    x = smoothstep(0., 1., x); // Basic smoothing - Equivalent to: x*x*(3. - 2.*x).
    return lerp(x, x2, .15);
    
/*    
    // Repeat sine gradient.
    float s = sin(x + 6.283*offs + 0.);
    return s*.5 + .5;
    // Sine lerped with an absolute sine wave.
    //float sa = sin((x +  6.283*offs)/2.);
    //return lerp(s*.5 + .5, 1. - abs(sa), .5);
    
*/
}

// One sand function layer... which is comprised of two lerped, rotated layers of repeat gradients lines.
float sandL(float2 p){
    
    // Layer one. 
    float2 q = mul(rot2(3.14159/18.),p); // Rotate the layer, but not too much.
    q.y += (gradN2D(q*18.) - .5)*.05; // Perturb the lines to make them look wavy.
    float grad1 = grad(q.y*80., 0.); // Repeat gradient lines.
   
    q = mul(rot2(-3.14159/20.),p); // Rotate the layer back the other way, but not too much.
    q.y += (gradN2D(q*12.) - .5)*.05; // Perturb the lines to make them look wavy.
    float grad2 = grad(q.y*80., .5); // Repeat gradient lines.
      
    
    // lerp the two layers above with an underlying 2D function. The function you choose is up to you,
    // but it's customary to use noise functions. However, in this case, I used a transcendental 
    // combination, because I like the way it looked better.
    // 
    // I feel that rotating the underlying lerping layers adds a little variety. Although, it's not
    // completely necessary.
    q = mul(rot2(3.14159/4.),p);
    //float c = lerp(grad1, grad2, smoothstep(.1, .9, n2D(q*float2(8,8))));//smoothstep(.2, .8, n2D(q*8.))
    //float c = lerp(grad1, grad2, n2D(q*float2(6,6)));//smoothstep(.2, .8, n2D(q*8.))
    //float c = lerp(grad1, grad2, dot(sin(q*12. - cos(q.yx*12.)), float2(.25,.25)) + .5);//smoothstep(.2, .8, n2D(q*8.))
    
    // The lerpes above will work, but I wanted to use a subtle screen blend of grad1 and grad2.
    float a2 = dot(sin(q*12. - cos(q.yx*12.)), float2(.25,.25)) + .5;
    float a1 = 1. - a2;
    
    // Screen blend.
    float c = 1. - (1. - grad1*a1)*(1. - grad2*a2);
    
    // Smooth max\min
    //float c = smax(grad1*a1, grad2*a2, .5);
   
    return c;
    
    
}

// A global value to record the distance from the camera to the hit point. It's used to tone
// down the sand height values that are further away. If you don't do this, really bad
// Moire artifacts will arise. By the way, you should always avoid globals, if you can, but
// I didn't want to pass an extra variable through a bunch of different functions.
float gT;

float sand(float2 p){
    
    // Rotating by 45 degrees. I thought it looked a little better this way. Not sure why.
    // I've also zoomed in by a factor of 4.
    p = float2(p.y - p.x, p.x + p.y)*.7071/4.;
    
    // Sand layer 1.
    float c1 = sandL(p);
    
    // Second layer.
    // Rotate, then increase the frequency -- The latter is optional.
    float2 q = mul(rot2(3.14159/12.),p);
    float c2 = sandL(q*1.25);
    
    // lerp the two layers with some underlying gradient noise.
    c1 = lerp(c1, c2, smoothstep(.1, .9, gradN2D(p*float2(4,4))));
    
/*   
    // Optional screen blending of the layers. I preferred the lerp method above.
    float a2 = gradN2D(p*float2(4,4));
    float a1 = 1. - a2;
    
    // Screen blend.
    c1 = 1. - (1. - c1*a1)*(1. - c2*a2);
*/    
    
    // Extra grit. Not really necessary.
    //c1 = .7 + fBm(p*128.)*.3;
    
    // A surprizingly simple and efficient hack to get rid of the super annoying Moire pattern 
    // formed in the distance. Simply lessen the value when it's further away. Most people would
    // figure this out pretty quickly, but it took me far too long before it hit me. :)
    return c1/(1. + gT*gT*.015);
}

/////////


// The path is a 2D sinusoid that varies over time, which depends upon the frequencies and amplitudes.
float2 path(in float z){ 
    
    return float2(4.*sin(z * .1), 0);
}

// The standard way to produce "cheap" dunes is to apply a triangle function to individual
// noise layers varying in amplitude and frequency. However, I needed something more subtle
// and rounder, so I've only applied a triangle function to the middle layer.
// 
// Here's an example using a more standard routine that's worth taking a look at:
//
// desert - wachel
// https://www.shadertoy.com/view/ltcGDl
float surfFunc( in float3 p){
    
    p /= 2.5;
    
    // Large base ampltude with lower frequency.
    float layer1 = n2D(p.xz*.2)*2. - .5; // Linear-like discontinuity - Gives an edge look.
    layer1 = smoothstep(0., 1.05, layer1); // Smoothing the sharp edge.

    // Medium amplitude with medium frequency. 
    float layer2 = n2D(p.xz*.275);
    layer2 = 1. - abs(layer2 - .5)*2.; // Triangle function, to give the dune edge look.
    layer2 = smoothstep(.2, 1., layer2*layer2); // Smoothing the sharp edge.
    
    // Smaller, higher frequency layer.
    float layer3 = n2D(p.xz*.5*3.);

     // Combining layers fBm style. Ie; Amplitudes inversely proportional to frequency.
    float res = layer1*.7 + layer2*.25 + layer3*.05;
    //float res = 1. - (1. - layer1*.7)*(1. - layer2*.25)*(1. - layer3*.05); // Screen 
    //float res = layer1*.75 + layer2*.25;

    return res;
    
}


// A similar -- trimmed down and smoothed out -- version of function above, for camera path usage.
float camSurfFunc( in float3 p){
    
    p /= 2.5;
    
    // Large base ampltude with lower frequency.
    float layer1 = n2D(p.xz*.2)*2. - .5; // Linear-like discontinuity - Gives an edge look.
    layer1 = smoothstep(0., 1.05, layer1); // Smoothing the sharp edge.

    // Medium amplitude with medium frequency. 
    float layer2 = n2D(p.xz*.275);
    layer2 = 1. - abs(layer2 - .5)*2.; // Triangle function, to give the dune edge look.
    layer2 = smoothstep(.2, 1., layer2*layer2); // Smoothing the sharp edge.

     // Combining layers fBm style. Ie; Amplitudes inversely proportional to frequency.
    float res = (layer1*.7 + layer2*.25)/.95;
    //float res = 1. - (1. - layer1*.75)*(1. - layer2*.25); // Screen 

    return res;
    
}



// The desert scene. Adding a heightmap to an XZ plane. Not a complicated distance function. :)
float map(float3 p){
    
    // Height map to perturb the flat plane. On a side note, I'll usually keep the
    // surface function within a zero to one range, which means I can use it later
    // for a bit of shading, etc. Of course, I could cut things down a bit, but at
    // the expense of confusion elsewhere... if that makes any sense. :)
    float sf = surfFunc(p);

    // Add the height map to the plane.
    return p.y + (.5-sf)*2.; 
 
}



// Basic raymarcher.
float trace(in float3 ro, in float3 rd){

    float t = 0., h;
    
    for(int i=0; i<96; i++){
    
        h = map(ro + rd*t);
        // Note the "t*b + a" addition. Basically, we're putting less emphasis on accuracy, as
        // "t" increases. It's a cheap trick that works in most situations... Not all, though.
        if(abs(h)<0.001*(t*.125 + 1.) || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.), etc.
        
        t += h; 
    }

    return min(t, FAR);
}

/*
// Tetrahedral normal - courtesy of IQ. I'm in saving mode, so the two "map" calls saved make
// a difference. Also because of the random nature of the scene, the tetrahedral normal has the 
// same aesthetic effect as the regular - but more expensive - one, so it's an easy decision.
float3 normal(in float3 p)
{  
    float2 e = float2(-1., 1.)*0.001;   
    return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
                     e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}
*/

 
// Standard normal function. It's not as fast as the tetrahedral calculation, but more symmetrical.
float3 normal(in float3 p, float ef) {
    float2 e = float2(0.001*ef, 0);
    return normalize(float3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
}

/*
// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch01.html
float3 tex3D(sampler2D t, in float3 p, in float3 n ){
    
    n = max(abs(n) - .2, 0.001);
    n /= dot(n, float3(1));
    float3 tx = texture(t, p.yz).xyz;
    float3 ty = texture(t, frac(p.zx)).xyz;
    float3 tz = texture(t, p.xy).xyz;

    // Textures are stored in sRGB (I think), so you have to convert them to linear space 
    // (squaring is a rough approximation) prior to working with them... or something like that. :)
    // Once the final color value is gamma corrected, you should see correct looking colors.
    return (tx*tx*n.x + ty*ty*n.y + tz*tz*n.z);
    
}


// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total.
float3 doBumpMap( sampler2D tx, in float3 p, in float3 n, float bf){
   
    const float2 e = float2(0.001, 0);
    
    // Three gradient floattors rolled into a matrix, constructed with offset greyscale texture values.    
    float3x3 m = float3x3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    float3 g = float3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), float3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}
*/

// Compact, self-contained version of IQ's 3D value noise function. I have a transparent noise
// example that explains it, if you require it.
float n3D_orig(in float3 p){
    
    const float3 s = float3(113, 157, 1);
    float3 ip = floor(p); p -= ip; 
    float4 h = float4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); //p *= p*p*(p*(p * 6. - 15.) + 10.);
    h = lerp(frac(sin(h)*43758.5453), frac(sin(h + s.x)*43758.5453), p.x);
    h.xy = lerp(h.xz, h.yw, p.y);
    return lerp(h.x, h.y, p.z); // Range: [0, 1].
}

// More concise, self contained version of IQ's original 3D noise function.
float n3D(in float3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
    const float3 s = float3(7, 157, 113);
    
    float3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride floattor for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    float4 h = float4(0., s.yz, s.y + s.z) + dot(ip, s);
    
    p -= ip; // Cell's fracional component.
    
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: frac(sin(x)*largeNumber).
    h = lerp(frac(sin(h)*43758.5453), frac(sin(h + s.x)*43758.5453), p.x);
    
    // Interpolating along Y.
    h.xy = lerp(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return lerp(h.x, h.y, p.z); // Range: [0, 1].
    
}


// 3D noise fBm.
float fBm(in float3 p){
    
    return n3D(p)*.57 + n3D(p*2.)*.28 + n3D(p*4.)*.15;
    
}


// Surface bump function..
float bumpSurf3D( in float3 p){
    
    // Obtaining some terrain samples in order to produce a gradient
    // with which to distort the sand. Basically, it'll make it look
    // like the underlying terrain it effecting the sand. The downside
    // is the three extra taps per bump tap... Ouch. :) Actually, it's
    // not that bad, but I might attempt to come up with a better way.
    float n = surfFunc(p);
    float3 px = p + float3(.001, 0, 0);
    float nx = surfFunc(px);
    float3 pz = p + float3(0, 0, .001);
    float nz = surfFunc(pz);
    
    // The wavy sand, that has been perturbed by the underlying terrain.
    return sand(p.xz + float2(n - nx, n - nz)/.001*1.);

}

// Standard function-based bump mapping routine: This is the cheaper four tap version. There's
// a six tap version (samples taken from either side of each axis), but this works well enough.
float3 doBumpMap(in float3 p, in float3 nor, float bumpfactor){
    
    // Larger sample distances give a less defined bump, but can sometimes lessen the aliasing.
    const float2 e = float2(0.001, 0); 
    
    // Gradient floattor: float3(df/dx, df/dy, df/dz);
    float ref = bumpSurf3D(p);
    float3 grad = (float3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx)) - ref)/e.x; 
    
    /*
    // Six tap version, for comparisson. No discernible visual difference, in a lot of cases.
    float3 grad = float3(bumpSurf3D(p - e.xyy) - bumpSurf3D(p + e.xyy),
                     bumpSurf3D(p - e.yxy) - bumpSurf3D(p + e.yxy),
                     bumpSurf3D(p - e.yyx) - bumpSurf3D(p + e.yyx))/e.x*.5;
    */
       
    // Adjusting the tangent floattor so that it's perpendicular to the normal. It's some kind 
    // of orthogonal space fix using the Gram-Schmidt process, or something to that effect.
    grad -= nor*dot(nor, grad);          
         
    // Applying the gradient floattor to the normal. Larger bump factors make things more bumpy.
    return normalize(nor + grad*bumpfactor);
    
}

// Cheap shadows are the bain of my raymarching existence, since trying to alleviate artifacts is an excercise in
// futility. In fact, I'd almost say, shadowing - in a setting like this - with limited  iterations is impossible... 
// However, I'd be very grateful if someone could prove me wrong. :)
float softShadow(float3 ro, float3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable.
    const int maxIterationsShad = 24; 
    
    float3 rd = lp - ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = 0.0015;  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

         
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        h = clamp(h, .1, .5); // max(h, .02);//
        dist += h;

        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (shade<.001 || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also, just for kicks. :)
    return min(max(shade, 0.) + .05, 1.); 
}



// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float calcAO(in float3 p, in float3 n)
{
    float ao = 0.0, l;
    const float maxDist = 4.;
    const float nbIte = 5.;
    //const float falloff = .9;
    for( float i=1.; i< nbIte+.5; i++ ){
    
        l = (i + .0)*.5/nbIte*maxDist;        
        ao += (l - map( p + n*l )); // / pow(1.+l, falloff);
    }
    
    return clamp(1.- ao/nbIte, 0., 1.);
}



// Standard sky routine: Gradient with sun and overhead cloud plane. I debated over whether to put more 
// effort in, but the dust is there and I'm saving cycles. I originally included sun flares, but wasn't 
// feeling it, so took them out. I might tweak them later, and see if I can make them work with the scene.
float3 getSky(float3 ro, float3 rd, float3 ld){ 
    
    // Sky color gradients.
    float3 col = float3(.8, .7, .5), col2 = float3(.4, .6, .9);
    
    //return lerp(col, col2, pow(max(rd.y*.5 + .9, 0.), 5.));  // Probably a little too simplistic. :)
     
    // lerp the gradients using the Y value of the unit direction ray. 
    float3 sky = lerp(col, col2, pow(max(rd.y + .15, 0.), .5));
    sky *= float3(.84, 1, 1.17); // Adding some extra vibrancy.
     
    float sun = clamp(dot(ld, rd), 0., 1.);
    sky += float3(1, .7, .4)*float3(pow(sun, 16.),pow(sun, 16.),pow(sun, 16.))*.2; // Sun flare, of sorts.
    sun = pow(sun, 32.); // Not sure how well GPUs handle really high powers, so I'm doing it in two steps.
    sky += float3(1, .9, .6)*float3(pow(sun, 32.),pow(sun, 32.),pow(sun, 32.))*.35; // Sun.
    
     // Subtle, fake sky curvature.
    rd.z *= 1. + length(rd.xy)*.15;
    rd = normalize(rd);
   
    // A simple way to place some clouds on a distant plane above the terrain -- Based on something IQ uses.
    const float SC = 1e5;
    float t = (SC - ro.y - .15)/(rd.y + .15); // Trace out to a distant XZ plane.
    float2 uv = (ro + t*rd).xz; // UV coordinates.
    
    // lerp the sky with the clouds, whilst fading out a little toward the horizon (The rd.y bit).
    if(t>0.) sky =  lerp(sky, float3(2,2,2), smoothstep(.45, 1., fBm(1.5*uv/SC))*
                        smoothstep(.45, .55, rd.y*.5 + .5)*.4);
    
    // Return the sky color.
    return sky;
}



// More concise, self contained version of IQ's original 3D noise function.
float noise3D(in float3 p){
    
    // Just some random figures, analogous to stride. You can change this, if you want.
    const float3 s = float3(113, 157, 1);
    
    float3 ip = floor(p); // Unique unit cell ID.
    
    // Setting up the stride floattor for randomization and interpolation, kind of. 
    // All kinds of shortcuts are taken here. Refer to IQ's original formula.
    float4 h = float4(0., s.yz, s.y + s.z) + dot(ip, s);
    
    p -= ip; // Cell's fracional component.
    
    // A bit of cubic smoothing, to give the noise that rounded look.
    p = p*p*(3. - 2.*p);
    
    // Standard 3D noise stuff. Retrieving 8 random scalar values for each cube corner,
    // then interpolating along X. There are countless ways to randomize, but this is
    // the way most are familar with: frac(sin(x)*largeNumber).
    h = lerp(frac(sin(h)*43758.5453), frac(sin(h + s.x)*43758.5453), p.x);
    
    // Interpolating along Y.
    h.xy = lerp(h.xz, h.yw, p.y);
    
    // Interpolating along Z, and returning the 3D noise value.
    return lerp(h.x, h.y, p.z); // Range: [0, 1].
    
}

/////
// Code block to produce some layers of smokey haze. Not sophisticated at all.
// If you'd like to see a much more sophisticated version, refer to Nitmitz's
// Xyptonjtroz example. Incidently, I wrote this off the top of my head, but
// I did have that example in mind when writing this.

// Hash to return a scalar value from a 3D floattor.
float hash31(float3 p){ return frac(sin(dot(p, float3(127.1, 311.7, 74.7)))*43758.5453); }

// Several layers of cheap noise to produce some subtle smokey haze.
// Start at the ray origin, then take some samples of noise between it
// and the surface point. Apply some very simplistic lighting along the 
// way. It's not particularly well thought out, but it doesn't have to be.
float getMist(in float3 ro, in float3 rd, in float3 lp, in float t){

    float mist = 0.;
    
    //ro -= float3(0, 0, _Time.y*3.);
    
    float t0 = 0.;
    
    for (int i = 0; i<24; i++){
        
        // If we reach the surface, don't accumulate any more values.
        if (t0>t) break; 
        
        // Lighting. Technically, a lot of these points would be
        // shadowed, but we're ignoring that.
        float sDi = length(lp-ro)/FAR; 
        float sAtt = 1./(1. + sDi*.25);
        
        // Noise layer.
        float3 ro2 = (ro + rd*t0)*2.5;
        float c = noise3D(ro2)*.65 + noise3D(ro2*3.)*.25 + noise3D(ro2*9.)*.1;
        //float c = noise3D(ro2)*.65 + noise3D(ro2*4.)*.35; 

        float n = c;//max(.65-abs(c - .5)*2., 0.);//smoothstep(0., 1., abs(c - .5)*2.);
        mist += n*sAtt;
        
        // Advance the starting point towards the hit point. You can 
        // do this with constant jumps (FAR/8., etc), but I'm using
        // a variable jump here, because it gave me the aesthetic 
        // results I was after.
        t0 += clamp(c*.25, .1, 1.);
        
    }
    
    // Add a little noise, then clamp, and we're done.
    return max(mist/48., 0.);
    
    // A different variation (float n = (c. + 0.);)
    //return smoothstep(.05, 1., mist/32.);

}

//////


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

    // Screen coordinates.
    float2 u = fragCoord; //(fragCoord - iResolution.xy*.5)/iResolution.y;
    
    // Camera Setup.     
    //float3 ro = float3(0, 1.2, _Time.y*2.); // Camera position, doubling as the ray origin.
    float3 lookAt = ro + float3(0, -.15, .5);  // "Look At" position.
    
    
    // Using the Z-value to perturb the XY-plane.
    // Sending the camera and "look at" floattors down the tunnel. The "path" function is 
    // synchronized with the distance function.
//    ro.xy += path(ro.z);
//    lookAt.xy += path(lookAt.z);
    
    // Raising the camera up and down with the terrain function and tilting it up or down
    // according to the slope. It's subtle, but it adds to the immersiveness of that mind 
    // blowing, endless-sand experience. :D
    float sfH = camSurfFunc(ro); 
    float sfH2 = camSurfFunc(lookAt); 
    float slope = (sfH2 - sfH)/length(lookAt - ro); // Used a few lines below.
    //slope = smoothstep(-.15, 1.15, (slope*.5 + .5)) - .5; // Smoothing the slope... Needs work.
     
    // Raising the camera with the terrain.
//    ro.y += sfH2; 
//    lookAt.y += sfH2;
    
 
    // Using the above to produce the unit ray-direction floattor.
    float FOV = 3.14159265/2.5; // FOV - Field of view.
    float3 forward = normalize(lookAt - ro);
    float3 right = normalize(float3(forward.z, 0, -forward.x )); 
    float3 up = cross(forward, right);

    // rd - Ray direction.
//    float3 rd = normalize(forward + FOV*u.x*right + FOV*u.y*up);
    
    // Swiveling the camera about the XY-plane (from left to right) when turning corners.
    // Naturally, it's synchronized with the path in some kind of way.
//    rd.xy = mul(rot2( path(lookAt.z).x/96.),rd.xy);
    
    // Subtle up and down tilt, or camera pitch, if you prefer.
//    rd.yz = mul(rot2(-slope/3.),rd.yz);
    
    // Usually, you'd just make this a unit directional light, and be done with it, but I
    // like some of the angular subtleties of point lights, so this is a point light a
    // long distance away. Fake, and probably not advisable, but no one will notice.
    float3 lp = float3(FAR*.25, FAR*.25, FAR) + float3(0, 0, ro.z);
 

    // Raymarching.
    float t = trace(ro, rd);
    
    gT = t;
    
   
    // Sky. Only retrieving a single color this time.
    //float3 sky = getSky(rd);
    
    // The passage color. Can't remember why I set it to sky. I'm sure I had my reasons.
    float3 col = float3(0,0,0);
    
    // Surface point. "t" is clamped to the maximum distance, and I'm reusing it to render
    // the mist, so that's why it's declared in an untidy postion outside the block below...
    // It seemed like a good idea at the time. :)
    float3 sp = ro+t*rd; 
    
    float pathHeight = sp.y;//surfFunc(sp);// - path(sp.z).y; // Path height line, of sorts.
    
    // If we've hit the ground, color it up.
    if (t < FAR){
    
        
        float3 sn = normal(sp, 1.); // Surface normal. //*(1. + t*.125)
        
        // Light direction floattor. From the sun to the surface point. We're not performing
        // light distance attenuation, since it'll probably have minimal effect.
        float3 ld = lp - sp;
        float lDist = max(length(ld), 0.001);
        ld /= lDist; // Normalize the light direct floattor.
        
        lDist /= FAR; // Scaling down the distance to something workable for calculations.
        float atten = 1./(1. + lDist*lDist*.025);

        
        // Texture scale factor.        
        const float tSize = 1./8.;
        
        
        // Function based bump mapping.
        sn = doBumpMap(sp, sn, .07);///(1. + t*t/FAR/FAR*.25)
        
        // Texture bump mapping.
        float bf = .01;//(pathHeight + 5. < 0.)?  .05: .025;
        //sn = doBumpMap(iChannel0, sp*tSize, sn, bf/(1. + t/FAR));
        
        
        // Soft shadows and occlusion.
        float sh = softShadow(sp + sn*.002, lp, 6., t); 
        float ao = calcAO(sp, sn); // Ambient occlusion.
        
        // Add AO to the shadow. No science, but adding AO to things sometimes gives a bounced light look.
        sh = min(sh + ao*.25, 1.); 
        
        float dif = max( dot( ld, sn ), 0.0); // Diffuse term.
        float spe = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 5.); // Specular term.
        float fre = clamp(1.0 + dot(rd, sn), 0.0, 1.0); // Fresnel reflection term.
 
        // Schlick approximation. I use it to tone down the specular term. It's pretty subtle,
        // so could almost be aproximated by a constant, but I prefer it. Here, it's being
        // used to give a sandstone consistency... It "kind of" works.
        float Schlick = pow( 1. - max(dot(rd, normalize(rd + ld)), 0.), 5.0);
        float fre2 = lerp(.2, 1., Schlick);  //F0 = .2 - Hard clay... or close enough.
       
        // Overal global ambience. It's made up, but I figured a little occlusion (less ambient light
        // in the corners, etc) and reflectance would be in amongst it... Sounds good, anyway. :)
        float amb = ao*.35;// + fre*fre2*.2;
        

        
        // Give the sand a bit of a sandstone texture.
        col = lerp(float3(1, .95, .7), float3(.9, .6, .4), fBm(sp.xz*16.));
        col = lerp(col*1.4, col*.6, fBm(sp.xz*32. - .5));///(1. + t*t*.001)
        
       
        // Extra shading in the sand crevices.
        float bSurf = bumpSurf3D(sp);
        col *= bSurf*.75 + .5;
       
        
        // Lamest sand sprinkles ever. :)
        col = lerp(col*.7 + (hash(floor(sp*96.))*.7 + hash(floor(sp*192.))*.3)*.3, col, min(t*t/FAR, 1.));
        
        col *= float3(1.2, 1, .9); // Extra color -- Part of last minute adjustments.
        
        
        // Combining all the terms from above. Some diffuse, some specular - both of which are
        // shadowed and occluded - plus some global ambience. Not entirely correct, but it's
        // good enough for the purposes of this demonstation.        
        col = col*(dif + amb + float3(1, .97, .92)*fre2*spe*2.)*atten;
        
        
        // A bit of sky reflection. Not really accurate, but I've been using fake physics since the 90s. :)
        float3 refSky = getSky(sp, reflect(rd, sn), ld);
        col += col*refSky*.05 + refSky*fre*fre2*atten*.15; 
        
 
        // Applying the shadows and ambient occlusion.
        col *= sh*ao;

        //col = float3(ao);
    }
    
  
    // Combine the scene with the sky using some cheap volumetric substance.
    float dust = getMist(ro, rd, lp, t)*(1. - smoothstep(0., 1., pathHeight*.05));//(-rd.y + 1.);
    float3 gLD = normalize(lp - float3(0, 0, ro.z));
    float3 sky = getSky(ro, rd, gLD);//*lerp(1., .75, dust);
    //col = lerp(col, sky, min(t*t*1.5/FAR/FAR, 1.)); // Quadratic fade off. More subtle.
    col = lerp(col, sky, smoothstep(0., .95, t/FAR)); // Linear fade. Much dustier. I kind of like it.
    
    
    // Mild dusty haze... Not really sure how it fits into the physical situation, but I thought it'd
    // add an extra level of depth... or something. At this point I'm reminded of the "dog in a tie 
    // sitting at the computer" meme with the caption, "I have no idea what I'm doing." :D
    float3 mistCol = float3(1, .95, .9); // Probably, more realistic, but less interesting.
    //col += (lerp(col, mistCol, .66)*.66 + col*mistCol*1.)*dust;
    
    
    // Simulating sun scatter over the sky and terrain: IQ uses it in his Elevated example.
    col += float3(1., .6, .2)*pow( max(dot(rd, gLD), 0.), 16.)*.45;
    
    
    // Applying the mild dusty haze.
    col = col*.75 + (col + .25*float3(1.2, 1, .9))*mistCol*dust*1.5;
    //col *= 1.05;
    
    
    // Really artificial. Kind of cool, but probably a little too much.    
    //col *= float3(1.2, 1, .9);

    
    // Standard way to do a square vignette. Note that the maxium value value occurs at "pow(0.5, 4.) = 1./16," 
    // so you multiply by 16 to give it a zero to one range. This one has been toned down with a power
    // term to give it more subtlety.
//    u = fragCoord/iResolution.xy;
//    col = min(col, 1.)*pow( 16.*u.x*u.y*(1. - u.x)*(1. - u.y) , .0625);
 
    // Done.
    fragColor = float4(sqrt(clamp(col, 0., 1.)), 1);


                return fragColor;
            }

            ENDCG
        }
    }
}




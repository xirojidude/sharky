
Shader "Skybox/TerrainPolyhedron"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _SunDir ("Sun Dir", Ve ctor) = (-.11,.07,0.99,0) 
        _XYZPos ("XYZ Offset", Ve ctor) = (0, 15, -.25 ,0) 
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



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin



                return fragColor;
            }

            ENDCG
        }
    }
}
// The cubemap texture resultion.
#define cubemapRes vec2(1024)


// Reading into one of the cube faces, according to the face ID. To save on cycles,
// I'd hardcode the face you're after into all but the least costly of situations.
// This particular function is used just once for an update in the "CubeA" tab.
//
// The four cube sides - Left, back, right, front.
// NEGATIVE_X, POSITIVE_Z, POSITIVE_X, NEGATIVE_Z
// vec3(-.5, uv.yx), vec3(uv, .5), vec3(.5, uv.y, -uv.x), vec3(-uv.x, uv.y, -.5).
//
// Bottom and top.
// NEGATIVE_Y, POSITIVE_Y
// vec3(uv.x, -.5, uv.y), vec3(uv.x, .5, -uv.y).

vec4 tx1(samplerCube tx, vec2 p){    

    p = fract(p) - .5;
    return textureLod(tx,  vec3(.5, p.y, -p.x), 0.);
    //return texture(tx, vec3(.5, p.y, -p.x));
}

/*
vec4 tx(samplerCube tx, vec2 p, int id){    

    vec4 rTx;
    
    vec2 uv = fract(p) - .5;
    // It's important to snap to the pixel centers. The people complaining about
    // seam line problems are probably not doing this.
    //p = (floor(p*cubemapRes) + .5)/cubemapRes; 
    
    vec3[6] fcP = vec3[6](vec3(-.5, uv.yx), vec3(.5, uv.y, -uv.x), vec3(uv.x, -.5, uv.y),
                          vec3(uv.x, .5, -uv.y), vec3(-uv.x, uv.y, -.5), vec3(uv, .5));
 
    
    return texture(tx, fcP[id]);
}

*/


// If you want things to wrap, you need a wrapping scale. It's not so important
// here, because we're performing a wrapped blur. Wrapping is not much different
// to regular mapping. You just need to put "p = mod(p, gSc)" in the hash function
// for anything that's procedurally generated with random numbers. If you're using
// a repeat texture, then that'll have to wrap too.
vec3 gSc;


// Fabrice's concise, 2D rotation formula.
//mat2 rot2(float th){ vec2 a = sin(vec2(1.5707963, 0) + th); return mat2(a, -a.y, a.x); }
// Standard 2D rotation formula - Nimitz says it's faster, so that's good enough for me. :)
mat2 rot2(in float a){ float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }

// mat3 rotation... I did this in a hurry, but I think it's right. :)
mat3 rot(vec3 ang){
    
    vec3 c = cos(ang), s = sin(ang);

    return mat3(c.x*c.z - s.x*s.y*s.z, -s.x*c.y, -c.x*s.z - s.x*s.y*c.z,
                c.x*s.y*s.z + s.x*c.z, c.x*c.y, c.x*s.y*c.z - s.x*s.z,
                c.y*s.z, -s.y, c.y*c.z);
    
}


// IQ's float to float hash.
float hash11(float x){  return fract(sin(x)*43758.5453); }


// IQ's vec2 to float hash.
float hash21(vec2 p){
    return fract(sin(dot(p, vec2(27.609, 157.583)))*43758.5453); 
}

/*
// IQ's unsigned box formula.
float sBoxSU(in vec2 p, in vec2 b, in float sf){

  return length(max(abs(p) - b + sf, 0.)) - sf;
}
*/

// IQ's signed box formula.
float sBoxS(in vec2 p, in vec2 b, in float sf){

  //return length(max(abs(p) - b + sf, 0.)) - sf;
  p = abs(p) - b + sf;
  return length(max(p, 0.)) + min(max(p.x, p.y), 0.) - sf;
}


// Commutative smooth maximum function. Provided by Tomkh, and taken 
// from Alex Evans's (aka Statix) talk: 
// http://media.lolrus.mediamolecule.com/AlexEvans_SIGGRAPH-2015.pdf
// Credited to Dave Smith @media molecule.
float smax(float a, float b, float k){
    
   float f = max(0., 1. - abs(b - a)/k);
   return max(a, b) + k*.25*f*f;
}


// Commutative smooth minimum function. Provided by Tomkh, and taken 
// from Alex Evans's (aka Statix) talk: 
// http://media.lolrus.mediamolecule.com/AlexEvans_SIGGRAPH-2015.pdf
// Credited to Dave Smith @media molecule.
float smin(float a, float b, float k){

   float f = max(0., 1. - abs(b - a)/k);
   return min(a, b) - k*.25*f*f;
}

/*
// IQ's exponential-based smooth maximum function. Unlike the polynomial-based
// smooth maximum, this one is associative and commutative.
float smaxExp(float a, float b, float k){

    float res = exp(k*a) + exp(k*b);
    return log(res)/k;
}
*/

// IQ's exponential-based smooth minimum function. Unlike the polynomial-based
// smooth minimum, this one is associative and commutative.
float sminExp(float a, float b, float k){

    float res = exp(-k*a) + exp(-k*b);
    return -log(res)/k;
}


// With the spare cycles, I thought I'd splash out and use Dave's more reliable hash function. :)
//
// Dave's hash function. More reliable with large values, but will still eventually break down.
//
// Hash without Sine.
// Creative Commons Attribution-ShareAlike 4.0 International Public License.
// Created by David Hoskins.
// vec3 to vec3.
vec3 hash33G(vec3 p){

    
    p = mod(p, gSc);
    p = fract(p * vec3(.10313, .10307, .09731));
    p += dot(p, p.yxz + 19.1937);
    p = fract((p.xxy + p.yxx)*p.zyx)*2. - 1.;
    return p;
   
    /*
    // Note the "mod" call. Slower, but ensures accuracy with large time values.
    mat2  m = rot2(mod(iTime, 6.2831853));  
    p.xy = m * p.xy;//rotate gradient vector
    p.yz = m * p.yz;//rotate gradient vector
    //p.zx = m * p.zx;//rotate gradient vector
    return p;
    */

}

/*
// Cheap vec3 to vec3 hash. I wrote this one. It's much faster than others, but I don't trust
// it over large values.
vec3 hash33(vec3 p){ 
   
    
    p = mod(p, gSc);
    //float n = sin(dot(p, vec3(7, 157, 113)));    
    //p = fract(vec3(2097152, 262144, 32768)*n)*2. - 1.; 
    
    //mat2  m = rot2(iTime);//in general use 3d rotation
    //p.xy = m * p.xy;//rotate gradient vector
    ////p.yz = m * p.yz;//rotate gradient vector
    ////p.zx = m * p.zx;//rotate gradient vector
    //return p;
    
    float n = sin(dot(p, vec3(57, 113, 27)));    
    return fract(vec3(2097152, 262144, 32768)*n)*2. - 1.;  

    
    //float n = sin(dot(p, vec3(7, 157, 113)));    
    //p = fract(vec3(2097152, 262144, 32768)*n); 
    //return sin(p*6.2831853 + iTime)*.5; 
}
*/

// hash based 3d value noise
vec4 hash41T(vec4 p){
    p = mod(p, vec4(gSc, gSc));
    return fract(sin(p)*43758.5453);
}

// Compact, self-contained version of IQ's 3D value noise function.
float n3DT(vec3 p){
    
    const vec3 s = vec3(27, 111, 57);
    vec3 ip = floor(p); p -= ip; 
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); 
    //p *= p*p*(p*(p*6. - 15.) + 10.);
    h = mix(hash41T(h), hash41T(h + s.x), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z); // Range: [0, 1].
}


// David_Hoskins puts together some pretty reliable hash functions. This is 
// his unsigned integer based vec3 to vec3 version.
vec3 hash33(vec3 p)
{
    p = mod(p, gSc);
    uvec3 q = uvec3(ivec3(p))*uvec3(1597334673U, 3812015801U, 2798796415U);
    q = (q.x ^ q.y ^ q.z)*uvec3(1597334673U, 3812015801U, 2798796415U);
    return -1. + 2. * vec3(q) * (1. / float(0xffffffffU));
}


// IQ's extrusion formula.
float opExtrusion(in float sdf, in float pz, in float h){
    
    vec2 w = vec2(sdf, abs(pz) - h);
    return min(max(w.x, w.y), 0.) + length(max(w, 0.));

    /*
    // Slight rounding. A little nicer, but slower.
    const float sf = .002;
    vec2 w = vec2( sdf, abs(pz) - h - sf/2.);
    return min(max(w.x, w.y), 0.) + length(max(w + sf, 0.)) - sf;
    */
}

// Signed distance to a regular hexagon -- using IQ's more exact method.
float sdHexagon(in vec2 p, in float r){
    
  const vec3 k = vec3(-.8660254, .5, .57735); // pi/6: cos, sin, tan.

  // X and Y reflection.
  p = abs(p);
  p -= 2.*min(dot(k.xy, p), 0.)*k.xy;
    
  // Polygon side.
  return length(p - vec2(clamp(p.x, -k.z*r, k.z*r), r))*sign(p.y - r);
    
}

/*
// vec2 to vec2 hash.
vec2 hash22G(vec2 p) { 

    p = mod(p, gSc.xy);
    // Faster, but doesn't disperse things quite as nicely. However, when framerate
    // is an issue, and it often is, this is a good one to use. Basically, it's a tweaked 
    // amalgamation I put together, based on a couple of other random algorithms I've 
    // seen around... so use it with caution, because I make a tonne of mistakes. :)
    float n = sin(dot(p, vec2(41, 289)));
    return fract(vec2(262144, 32768)*n)*2. - 1.; 
    
    // Animated.
    //p = fract(vec2(262144, 32768)*n); 
    // Note the ".45," insted of ".5" that you'd expect to see. When edging, it can open 
    // up the cells ever so slightly for a more even spread. In fact, lower numbers work 
    // even better, but then the random movement would become too restricted. Zero would 
    // give you square cells.
    //return sin( p*6.2831853 + iTime ); 
    
}

// return gradient noise (in x) and its derivatives (in yz)
vec3 n2D3G( in vec2 p )
{
   
    vec2 i = floor( p );
    vec2 f = p - i;

#if 1
    // quintic interpolation
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
#else
    // cubic interpolation
    vec2 u = f*f*(3.0-2.0*f);
    vec2 du = 6.0*f*(1.0-f);
#endif    
    
    vec2 ga = hash22G( i + vec2(0.0,0.0) );
    vec2 gb = hash22G( i + vec2(1.0,0.0) );
    vec2 gc = hash22G( i + vec2(0.0,1.0) );
    vec2 gd = hash22G( i + vec2(1.0,1.0) );
    
    float va = dot( ga, f - vec2(0.0,0.0) );
    float vb = dot( gb, f - vec2(1.0,0.0) );
    float vc = dot( gc, f - vec2(0.0,1.0) );
    float vd = dot( gd, f - vec2(1.0,1.0) );

    return vec3( va + u.x*(vb-va) + u.y*(vc-va) + u.x*u.y*(va-vb-vc+vd),   // value
                 ga + u.x*(gb-ga) + u.y*(gc-ga) + u.x*u.y*(ga-gb-gc+gd) +  // derivatives
                 du * (u.yx*(va-vb-vc+vd) + vec2(vb,vc) - va))*.5 + .5;
}
*/

// IQ's vec2 to float hash.
float hash21T(vec2 p){
    p = mod(p, gSc.xy);
    return fract(sin(dot(p, vec2(27.609, 157.583)))*43758.5453); 
}


// Value noise, and its analytical derivatives -- Courtesy of IQ.
vec3 n2D3( in vec2 x )
{
    vec2 p = floor(x);
    vec2 f = x - p;
    
// cubic interpolation vs quintic interpolation
#if 0 
    vec2 u = f*f*(3.0-2.0*f);
    vec2 du = 6.0*f*(1.0-f);
    //vec2 ddu = 6.0 - 12.0*f;
#else
    vec2 u = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 du = 30.0*f*f*(f*(f-2.0)+1.0);
    //vec2 ddu = 60.0*f*(1.0+f*(-3.0+2.0*f));
#endif
    
    float a = hash21T(p);//textureLod(iChannel0,(p+vec2(0.5,0.5))/256.0,0.0).x;
    float b = hash21T(p+vec2(1,0));
    float c = hash21T(p+vec2(0,1));
    float d = hash21T(p+1.);
    
    float k0 =   a;
    float k1 =   b - a;
    float k2 =   c - a;
    float k4 =   a - b - c + d;


    // value
    float va = k0+k1*u.x+k2*u.y+k4*u.x*u.y;
    // derivative                
    vec2  de = du*(vec2(k1,k2)+k4*u.yx);
    // hessian (second derivartive)
   /* mat2  he = mat2( ddu.x*(k1 + k4*u.y),   
                     du.x*k4*du.y,
                     du.y*k4*du.x,
                     ddu.y*(k2 + k4*u.x) );*/
    
    return vec3(va,de);

}

// The terrain function. All layers are wrapped.
vec3 terrain(vec2 p){
 
    // Result, amplitude and sum.
    float res = 0., a = 1., sum = 0.;
    vec2 dfn = vec2(0); // Gradient.
    vec3 fn; // Function.
 
    for (int i=0; i<5; i++){
        fn = n2D3(p); // Function.
        dfn += fn.yz; // Gradient.
        // Tempering the layers with the gradient.
        res += fn.x*a/(1. + dot(dfn, dfn)*.5); //(1.-abs(fn.x - .5)*2.)
        //p = mat2(1.6,-1.2,1.2,1.6)*p; // r2(3.14159/7.5)*p*2.; //
        p *= 2.;// Increasing frequency.
        gSc = min(gSc*2., 1024.); // Wrapping the increased frequency.
        sum += a; // Sum total. ///(1. + dot(dfn, dfn)*.5);
        a *= .5; // Decreasing the amplitude.
    }
     
    res /= sum; // Height value.
    
    //res = res*.5 + .5;

    
    return vec3(res, dfn.xy);
    
}




// This is a variation on a regular 2-pass Voronoi traversal that produces a Voronoi
// pattern based on the interior cell point to the nearest cell edge (as opposed
// to the nearest offset point). It's a slight reworking of Tomkh's example, which
// in turn, is based on IQ's original example. The links are below:
//
// On a side note, I have no idea whether a faster solution is possible, but when I
// have time, I'm going to try to find one anyway.
//
// Voronoi distances - iq
// https://www.shadertoy.com/view/ldl3W8
//
// Here's IQ's well written article that describes the process in more detail.
// http://www.iquilezles.org/www/articles/voronoilines/voronoilines.htm
//
// Faster Voronoi Edge Distance - tomkh
// https://www.shadertoy.com/view/llG3zy
//
//
vec3 cellID;
int gIFrame;

ivec4 gID;

// Distance metric: Put whatever you want here.
float distMetric(vec3 p, vec3 b, int id){
    
    
    if(id==0){
        return (dot(p, p));///2.; // Sphere squared.
    }
    else {
        
        //float d2 = sBoxS(p.xy, b.xy, 0.);
        float d2 = sdHexagon(p.xy, min(b.x, b.y));
        return opExtrusion(d2, p.z, b.z);

        
        //return (dot(p, p));
        //return length(p); // Standard spherical Euclidean distance.

        //return max(max(length(p.xy), length(p.yz)), length(p.xz)); // Cylinder cross.

        //p = max(abs(p)*.8660254 + p.yzx*.5, -p);
        //return max(max(p.x, p.y), p.z); // Triangular.

        // Uncomment this for all metrics below.
        p = abs(p) - b;
        
        
        //p = (p + p.yzx)*.7071;
        //return max(max(p.x, p.y), p.z); // Can't remember -- Diamond related. :)


        return max(max(p.x, p.y), p.z); // Cube.
        //return (p.x + p.y + p.z)*.5;//7735; // Octahedron.

        //p = p*.8660254 + p.yzx*.5;
        //return max(max(p.x, p.y), p.z); // Hex.

/*        
        float taper = (p.x + p.y + p.z)/3.*2.*.65 + .35; // Linear gradient of sorts.
        //float taper = p.y + .5; // Original.
        //taper = mix(taper, max(taper, .5), .35); // Flattening the sharp edge a bit.

        p = abs(p)*2.;
        //p = vec2(abs(p.x)*1.5, (p.y)*1.5 - .25)*2.; // Used with triangle.

        float shape = max(max(p.x, p.y), p.z); // Square.
        //float shape = max(p.x*.866025 - p.y*.5, p.y); // Triangle.
        //float shape = max(p.x*.866025 + p.y*.5, p.y); // Hexagon.
        //float shape = max(max(p.x, p.y), (p.x + p.y)*.7071); // Octagon.
        //float shape = length(p); // Circle.
        //float shape = dot(p, p); // Circle squared.


        //shape = (shape - .125)/(1. - .125);
        //shape = smoothstep(0., 1., shape);


        //return shape;
        return max(shape, taper);
*/
    }
    
}

// 2D 3rd-order Voronoi: This is just a rehash of Fabrice Neyret's version, which is in
// turn based on IQ's original. I've simplified it slightly, and tidied up the "if" statements.
//
vec3 Voronoi(in vec3 q, in vec3 sc, in vec3 rotF, float offsF, int id){
    
    
    //const vec3 sc = vec3(1, 2, 1);
    gSc /= sc;
    vec3 d = vec3(1e5); // 1.4, etc.
    
    float r;
    
    // Widen or tighten the grid coverage, depending on the situation. Note the huge (5x5x5 tap) 
    // spread. That's to cover the third order distances. In a lot of cases, (3x3x3) is enough,
    // but in some, 64 taps (4x4x4), or even more, might be necessary.
    //
    // Either way, this is fine for static imagery, but needs to be reined in for realtime use.
    for(int z = -2; z <= 2; z++){ 
        for(int y = -2; y <= 2; y++){ 
            for(int x =-2; x <= 2; x++){

                vec3 cntr = vec3(x, y, z) - .5;
                vec3 p = q;
                vec3 ip = floor(p/sc) + .5; 
                p -= (ip + cntr)*sc;
                ip += cntr;
                
                // Random position and rotation vectors.
                vec3 rndP = hash33(ip + 5.);
                vec3 rndR = hash33(ip + 7.)*6.2831*rotF;

                // Rotate.
                p = rot(rndR)*p;
                //p.xy *= rot2(rndR.x);
                //p.yz *= rot2(rndR.y);
                //p.zx *= rot2(rndR.z);
               
                // Postional offset.
                p -= rndP*offsF*sc;
                
                
                // Scale -- Redundant here.
                vec3 b = sc/2.*vec3(1, 1, 1);
                // Distance metric.
                r = distMetric(p, b, id);

                // 1st, 2nd and 3rd nearest distance metrics.
                d.z = max(d.x, max(d.y, min(d.z, r))); // 3rd.
                d.y = max(d.x, min(d.y, r)); // 2nd.
                d.x = min(d.x, r);//smin(d.x, r, .2); // Closest.
                
                // Redundant break in an attempt to ensure no unrolling.
                // No idea whether it works or not.
                if(d.x>1e5) break; 

            }
        }
    }

    
    return min(d, 1.);
    
}














// It can be a bit fiddly filling all four channels in at once, but thankfully, this is
// all calculated at startup. The idea is to put the function you wish to use in the
// middle of the loop here, instead of writing it out four times over.
vec4 funcFace1(vec2 uv){
    
    // It's a 2D conversion, but we're using a 3D function with constant Z value.
    vec3 p;
    // Just choose any Z value you like. You could actually set "p.z" to any constant,
    // or whatever, but I'm keeping things consistant.
    p.z = floor(.0*cubemapRes.x)/cubemapRes.x; 
       
    vec4 col;
    
    for(int i = 0; i<4; i++){

        // Since we're performing our own 2D interpolation, it makes sense to store
        // neighboring values in the other pixel channels. It makes things slightly
        // more confusing, but saves four texel lookups -- usually in the middel of
        // a raymarching loop -- later on.
        
        // The neighboring position for each pixel channel.
        p.xy = mod(floor(uv*cubemapRes) + vec2(i&1, i>>1), cubemapRes)/cubemapRes;
         
        // Layering in some noise as well. This is all precalculated, so speed isn't
        // the primary concern... Compiler time still needs to be considered though.
        gSc = vec3(4);
        float res2 = n3DT(p*gSc);
        gSc = vec3(8);
        res2 = mix(res2, n3DT(p*gSc), .333);
        gSc = vec3(16);
        res2 = mix(res2, n3DT(p*gSc), .333/2.);
        //gSc = vec3(32);
        //res2 = mix(res2, n3DT(p*gSc), .333/4.);
        gSc = vec3(64);
        res2 = mix(res2, 1. - abs(.5 - n3DT(p*gSc))*2., .02);
 
       

        // Individual Voronoi cell scaling.
        vec3 sc = vec3(1, 1, 1);
        vec3 rotF = vec3(0); // Rotation factor.
        
        //sc += res2*.05;
        
        // Put whatever function you want here. In this case, it's Voronoi.
        gSc = vec3(32);
        vec3 v = Voronoi(p*gSc, sc, rotF, 1., 0);
        float res = v.x;
        gSc = vec3(64);
        v = Voronoi(p*gSc, sc, rotF, 1., 0);
        res = mix(res, v.x, .333);
        gSc = vec3(256);
        v = Voronoi(p*gSc, sc, rotF, 1., 0);
        res = mix(res, mix(v.y - v.x, smoothstep(.1, 1., v.y - v.x), .5), .333/3.);
        
        
        
        // The pixel channel value: On a side note, setting it to "v.y" is interesting,
        // but not the look we're going for here.
        
        
        
        // Mix in the Voronoi and the noise.
        col[i] = mix(res, res2, .9);
        
        
        
        gSc = vec3(4);
        vec3 r3 = terrain(p.xy*gSc.xy + .5);
        float res3 = smoothstep(0.1, 1., r3.x);
        
        col[i] = mix(res, res3, .9);

    }
    
    return col;
}



// Cube mapping - Adapted from one of Fizzer's routines. 
int CubeFaceCoords(vec3 p){

    // Elegant cubic space stepping trick, as seen in many voxel related examples.
    vec3 f = abs(p); f = step(f.zxy, f)*step(f.yzx, f); 
    
    ivec3 idF = ivec3(p.x<.0? 0 : 1, p.y<.0? 2 : 3, p.z<0.? 4 : 5);
    
    return f.x>.5? idF.x : f.y>.5? idF.y : idF.z; 
} 

void mainCubemap(out vec4 fragColor, in vec2 fragCoord, in vec3 rayOri, in vec3 rayDir){
    
    
    // UV coordinates.
    //
    // For whatever reason (which I'd love expained), the Y coordinates flip each
    // frame if I don't negate the coordinates here -- I'm assuming this is internal, 
    // a VFlip thing, or there's something I'm missing. If there are experts out there, 
    // any feedback would be welcome. :)
    vec2 uv = fract(fragCoord/iResolution.y*vec2(1, -1));
    
    // Adapting one of Fizzer's old cube mapping routines to obtain the cube face ID 
    // from the ray direction vector.
    int faceID = CubeFaceCoords(rayDir);
  
  
    // Pixel storage.
    vec4 col;
    

    // Initial conditions -- Performed upon initiation.
    //if(abs(tx(iChannel0, uv, 5).w - iResolution.y)>.001){
    //if(iFrame<1){
    //
    // Great hack, by IQ, to ensure that this loads either on the first frame, or in the
    // event that the texture hasn't loaded (this happens a lot), wait, then do it...
    // Well kind of. Either way, it works. It's quite clever, which means that it's something 
    // I never would have considered. :)
    if(textureSize(iChannel0,0).x<2 || iFrame<1){
      
        
        /*
        // Debug information for testing individual cubeface access.
        if(faceID==0) col = vec4(0, 1, 0, 1);
        else if(faceID==1) col = vec4(0, .5, 1, 1);
        else if(faceID==2) col = vec4(1, 1, 0, 1);
        else if(faceID==3) col = vec4(1, 0, 0, 1);
        else if(faceID==4) col = vec4(.5, .5, .5, 1);
        else col = vec4(1, 1, 1, 1);
        */
 
        
        // Fill the second cube face with a custom 2D function... We're actually
        // reusing a 3D function, but it's in slice form, which essentially makes
        // it a 2D function.
        if(faceID==1){

            col = funcFace1(uv);
            
        } 
        
        /*
        // Last channel on the last face: Used to store the current 
        // resolution to ensure loading... Yeah, it's wasteful and it
        // slows things down, but until there's a reliable initiation
        // variable, I guess it'll have to do. :)
        if(faceID==5){
            
            col.w = iResolution.y;
        }
        */

        
    }
    else {
            
        // The cube faces have already been initialized with values, so from this point,
        // read the values out... There's probably a way to bypass this by using the 
        // "discard" operation, but this isn't too expensive, so I'll leave it for now.
        //col = tx(iChannel0, uv, faceID);
        if(faceID == 1) col = tx1(iChannel0, uv);
    }
    
    
    // Update the cubemap faces.
    fragColor = col;
    
}











/*

    Mirrored Polyhedron On Terrain
    ------------------------------

    I love looking at those simple but beautiful static geometric images that Blender 
    artists, and so forth, post on various forums. Obviously, we can't quite produce the 
    same in realtime, but it's actually pretty easy to emulate the spirit of the images. 
    This particular scene was easy to produce, and pretty cheap also.

    Since the sky takes up half the viewing space, I figured I'd better at least attempt
    to put a believable one in there. You'd be amazed at just how effective a simple blueish
    gradient can be at emulating atmospheric scattering, but sometimes it's not enough. 
    Physicists and graphics coders have some awesome routines out there, but codewise, it 
    tends to blow things out too much for a simple example. Therefore, I decided on a very 
    cheap scattering routine that does a pretty good job... provided you're not too picky. :)
    There's a lot of cheap scattering code on Shadertoy, but I've based this particular one 
    on Bearwork's effective and easy to use "Fast Atmospheric Scattering" example; The link 
    is below, for anyone interested.

    The default polyhedral object is a Pentakis icosahedron, which is built on Djinn Kahn's 
    framework -- I put it together in a hurry, so there'd be better ways to go about it,
    but it does the job. There's also an option to replace it with a polyhedron built on 
    Knighty's framework -- For anyone interested, I provided a brief explanation inside 
    the "map" function as to how it works.

    By the way, this example has been patched together from previous examples in a hurry,
    but I'll tidy up the code and comments in due course.
    


    Related examples:

    // Pretty cool. I'm going to post something along these at some stage.
    Reflective Polyhedra - Dr2
    https://www.shadertoy.com/view/3ljSDV

    // A very efficient way to make some basic polyhedral examples. It's one 
    // thing understanding how space folding works, and another trying to calculate
    // the initial vectors required to put it all together. Thankfully, Knighty has 
    // done all the hard work. :)
    Polyhedras with control - knighty
    https://www.shadertoy.com/view/MsKGzw

    // With some tweaks, this does a great job at emulating scattering without
    // having to drop a heap of code into your example.
    Fast Atmospheric Scattering - bearworks
    https://www.shadertoy.com/view/WtBXWw

    // There's a lot of atmospheric scattering examples on here, but this is
    // one of my favorite all rounders. I'd like to do an example with a tweaked
    // version of this later.
    Real-Time Atmospheric Scattering - rayferric
    https://www.shadertoy.com/view/wllyW4

    // I used a different, simpler model, but I like this approach.
    Simple atmospheric scattering - w23
    https://www.shadertoy.com/view/XsKfWz

*/

// The far plane. I'd like this to be larger, but the extra iterations required to render the 
// additional scenery starts to slow things down on my slower machine.
#define FAR 60.

// The far plane would be about 6 kilometers away and the sun is about 150 million kilometers
// away, so this would be the relative distance... Actually, I have no idea, but any large
// number will do. :D
#define FARSUN 1.5e9


// Minimum surface distance. Used in various calculations.
#define DELTA .001


// Ray passes: For this example, just one intersection and one reflection.
#define PASSES 2

// The default is a Pentakis icosahedron, which is built on Djinn Kahn's framework. 
// Commenting it out will replace it with a polyhedron built on Knighty's framework.
#define PENTAKIS_ICOSAHEDRON

// Global distance marker: Used to soften things. 
float gT;


// Tri-Planar blending function. Based on an old Nvidia tutorial by Ryan Geiss.
vec3 tex3D( sampler2D t, in vec3 p, in vec3 n ){ 
    
    n = n = max(abs(n) - .2, 0.001); // max(abs(n), 0.001), etc.
    n /= dot(n, vec3(1));
    vec3 tx = texture(t, p.yz).xyz;
    vec3 ty = texture(t, p.zx).xyz;
    vec3 tz = texture(t, p.xy).xyz;
    
    // Textures are stored in sRGB (I think), so you have to convert them to linear space 
    // (squaring is a rough approximation) prior to working with them... or something like that. :)
    // Once the final color value is gamma corrected, you should see correct looking colors.
    return (tx*tx*n.x + ty*ty*n.y + tz*tz*n.z);
}


/* 
// Standard 2x2 hash algorithm.
vec2 hash22(vec2 p) {
    
    // Faster, but probaly doesn't disperse things as nicely as other methods.
    float n = sin(dot(p, vec2(113, 1)));
    return fract(vec2(2097152, 262144)*n)*2. - 1.;

}
*/

// Dave's hash function. More reliable with large values, but will still eventually break down.
//
// Hash without Sine
// Creative Commons Attribution-ShareAlike 4.0 International Public License
// Created by David Hoskins.
// vec2 to vec2.
vec2 hash22(vec2 p){

    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 19.19);
    p = fract((p3.xx + p3.yz)*p3.zy)*2. - 1.;
    return p;
    
    // Note the "mod" call. Slower, but ensures accuracy with large time values.
    //mat2  m = r2(mod(iTime, 6.2831853)); 
    //p.xy = m * p.xy;//rotate gradient vector
    //return p;
}

// Gradient noise. Ken Perlin came up with it, or a version of it. Either way, this is
// based on IQ's implementation. It's a pretty simple process: Break space into squares, 
// attach random 2D vectors to each of the square's four vertices, then smoothly 
// interpolate the space between them.
float gradN2D(in vec2 f){
    
    // Used as shorthand to write things like vec3(1, 0, 1) in the short form, e.yxy. 
   const vec2 e = vec2(0, 1);
   
    // Set up the cubic grid.
    // Integer value - unique to each cube, and used as an ID to generate random vectors for the
    // cube vertiies. Note that vertices shared among the cubes have the save random vectors attributed
    // to them.
    vec2 p = floor(f);
    f -= p; // Fractional position within the cube.
    

    // Smoothing - for smooth interpolation. Use the last line see the difference.
    //vec2 w = f*f*f*(f*(f*6.-15.)+10.); // Quintic smoothing. Slower and more squarish, but derivatives are smooth too.
    vec2 w = f*f*(3. - 2.*f); // Cubic smoothing. 
    //vec2 w = f*f*f; w = ( 7. + (w - 7. ) * f ) * w; // Super smooth, but less practical.
    //vec2 w = .5 - .5*cos(f*3.14159); // Cosinusoidal smoothing.
    //vec2 w = f; // No smoothing. Gives a blocky appearance.
    
    // Smoothly interpolating between the four verticies of the square. Due to the shared vertices between
    // grid squares, the result is blending of random values throughout the 2D space. By the way, the "dot" 
    // operation makes most sense visually, but isn't the only metric possible.
    float c = mix(mix(dot(hash22(p + e.xx), f - e.xx), dot(hash22(p + e.yx), f - e.yx), w.x),
                  mix(dot(hash22(p + e.xy), f - e.xy), dot(hash22(p + e.yy), f - e.yy), w.x), w.y);
    
    // Taking the final result, and converting it to the zero to one range.
    return c*.5 + .5; // Range: [0, 1].
}

// Gradient noise fBm.
float fBm(in vec2 p){
    
    return gradN2D(p)*.57 + gradN2D(p*2.)*.28 + gradN2D(p*4.)*.15;
    
}


// The path is a 2D sinusoid that varies over time, which depends upon the frequencies and amplitudes.
vec2 path(in float z){ 
    return vec2(0);
    //return vec2(cos(z*.18/1.)*2. - sin(z*.1/1.)*4., (sin(z*.12/1.)*3. - 1.)*0.);
}


// A 2D texture lookup: GPUs don't make it easy for you. If wrapping wasn't a concern,
// you could get away with just one GPU-filtered filtered texel read. However, there
// are seam line issues, which means you need to interpolate by hand, so to speak.
// Thankfully, you can at least store the four neighboring values in one pixel channel,
// so you're left with one texel read and some simple interpolation.
//
// By the way, I've included the standard noninterpolated option for comparisson.
float txFace1(in samplerCube tx, in vec2 p){
   
    
    p *= cubemapRes;
    vec2 ip = floor(p); p -= ip;
    vec2 uv = fract((ip + .5)/cubemapRes) - .5;
    
    #if 0
    
    // The standard noninterpolated option. It's faster, but doesn't look very nice.
    // You could change the texture filtering to "mipmap," but that introduces seam
    // lines at the borders -- which is fine, if they're out of site, but not when you
    // want to wrap things, which is almost always.
    return texture(tx, vec3(.5, uv.y, -uv.x)).x; 
    
    #else
    
    // Smooth 2D texture interpolation using just one lookup. The pixels and
    // its three neighbors are stored in each channel, then interpolated using
    // the usual methods -- similar to the way in which smooth 2D noise is
    // created.
    vec4 p4 = texture(tx, vec3(.5, uv.y, -uv.x)); 

    return mix(mix(p4.x, p4.y, p.x), mix(p4.z, p4.w, p.x), p.y);
    
    // Returning the average of the neighboring pixels, for curiosity sake.
    // Yeah, not great. :)
    //return dot(p4, vec4(.25));
    
    #endif
/*   
    // Four texture looks ups. I realized later that I could precalculate all four of 
    // these, pack them into the individual channels of one pixel, then read them
    // all back in one hit, which is much faster.
    vec2 uv = fract((ip + .5)/cubemapRes) - .5;
    vec4 x = texture(tx, vec3(.5, uv.y, -uv.x)).x;
    uv = fract((ip + vec2(1, 0)+ .5)/cubemapRes) - .5;
    vec4 y = texture(tx, vec3(.5, uv.y, -uv.x)).x;
    uv = fract((ip + vec2(0, 1)+ .5)/cubemapRes) - .5;
    vec4 z = texture(tx, vec3(.5, uv.y, -uv.x)).x;
    uv = fract((ip + vec2(1, 1)+ .5)/cubemapRes) - .5;
    vec4 w = texture(tx, vec3(.5, uv.y, -uv.x)).x;

    return mix(mix(x, y, p.x), mix(z, w, p.x), p.y);
*/  
    
}

// 2D Surface function.
float surfFunc2D(in vec3 p){
    
     return txFace1(iChannel0, p.xz/16.);///(1. + gT*gT*.001);
}

 

// This is a trimmed down version of one of Gaz's clever routines. I find it a 
// lot cleaner than those functions full of a million trigonometric variables.
vec3 rot3(in vec3 p, vec3 a, float t){
    a = normalize(a);
    vec3 q = cross(a, p), u = cross(q, a);
    return mat3(u, q, a)*vec3(cos(t), sin(t), dot(p, a));
}

//////

#ifdef PENTAKIS_ICOSAHEDRON

// The following have come from DjinnKahn's "Icosahedron Weave" example, here:
// https://www.shadertoy.com/view/Xty3Dy
//
// Vertices: vec3(0, A, B), vec3(B, 0, A), vec3(-B, 0, A).
// Face center: (vec3(0, A, B) + vec3(0, 0, A)*2.)/3..
// Edges: (vec3(0, A, B) + vec3(B, 0, A))/2.,  etc.

const float PHI = (1. + sqrt(5.))/2.; // 1.618
const float A = PHI/sqrt(1. + PHI*PHI); // .85064
const float B = 1./sqrt( 1. + PHI*PHI); // .5257
const float J = (PHI - 1.)/2.; // .309016994375
const float K = PHI/2.; // J + .5
const mat3 R0 = mat3(.5,  -K,   J,  K,  J, -.5,  J , .5,  K);
const mat3 R1 = mat3( K,   J, -.5,  J, .5,   K, .5 , -K,  J);
const mat3 R2 = mat3(-J, -.5,   K, .5, -K,  -J,  K ,  J, .5);

// I wanted all vertices hardcoded. Everything's been projected to the
// surface of a sphere.
const float size = 1.;
const vec3 v0 = (vec3(0, A, B))*size; // Already normalized.
const vec3 v1 = (vec3(B, 0, A))*size;
const vec3 v2 = (vec3(-B, 0, A))*size;
const vec3 e0 = normalize(mix(v0, v1, .5))*size;
const vec3 e1 = normalize(mix(v1, v2, .5))*size;
const vec3 e2 = normalize(mix(v2, v0, .5))*size;
//const vec3 mid = normalize(vec3(0, A, B + A*2.))/3.; // (v0 + v1 + v2)/3.*size.

// The original function -- sans polarity information -- is neat and concise.
vec3 opIcosahedron(vec3 p){ 
  
    p = R0*abs(p);
    p = R1*abs(p);
    p = R2*abs(p); 
    return abs(p);  
} 


#else

// Setup constants from Knighty's Polyhedras with control example, here:
// https://www.shadertoy.com/view/MsKGzw
//
// By the way, the "type" variable can be changed to "3" and "4," for other polyhedral arrangements.
const float type = 5., cospin = cos(3.14159265/type), scospin = sqrt(.75 - cospin*cospin);
const vec3 nc = vec3(-.5, -cospin, scospin); // 3rd folding plane. The two others are XZ and YZ planes.
const vec3 pab = vec3(0, 0, 1);

#endif

///////



 
// IQ's 3D line segment formula. Simpler and cheaper, but doesn't orient carved cross-sections.
float sdCapsule(vec3 p, vec3 a, vec3 b){

    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    pa = abs(pa - ba*h);
    return length( pa );
}

float dot2( in vec3 v ) { return dot(v, v); }

// IQ's 3D triangle routine, which you can find here:
// Triangle - distance 3D  - https://www.shadertoy.com/view/4sXXRN
//
float udTriangle(in vec3 p, in vec3 v1, in vec3 v2, in vec3 v3)
{
     
    vec3 v21 = v2 - v1; vec3 p1 = p - v1;
    vec3 v32 = v3 - v2; vec3 p2 = p - v2;
    vec3 v13 = v1 - v3; vec3 p3 = p - v3;
    vec3 nor = cross( v21, v13 );

    return sqrt( (sign(dot(cross(v21, nor), p1)) + 
                  sign(dot(cross(v32, nor), p2)) + 
                  sign(dot(cross(v13, nor), p3))<2.) 
                  ?
                  min( min( 
                  dot2(v21*clamp(dot(v21, p1)/dot2(v21),0., 1.) - p1), 
                  dot2(v32*clamp(dot(v32, p2)/dot2(v32),0., 1.) - p2) ), 
                  dot2(v13*clamp(dot(v13, p3)/dot2(v13),0., 1.) - p3) )
                  :
                  dot(nor,p1)*dot(nor,p1)/dot2(nor));
 
     
}

vec3 moveBall(vec3 p){
    
    p -= vec3(0, 1, 0);
    return rot3(p, vec3(3.14159/12., 3.14159/6., 0), iTime/2.);
}


// Rock and object ID holders.
int rID = 0;
int svRID;
vec4 vRID;
vec4 svVRID;

// The desert scene. Adding a heightmap to an XZ plane. Not a complicated distance function. :)
float map(vec3 p){
 
    
    // Retrieve the 2D surface value from a cube map face.
    float sf2D = surfFunc2D(p);
    
     
    // Path function. Not used here.
    //vec2 pth = path(p.z); 

    
    // Tunnel. Not used here.
    //float tun = 1. - dist((p.xy - pth)*vec2(.7, 1));

    // Mover the mirrored ball object.
    vec3 q = moveBall(p);
    
     
    // Terrain.
    float ter = p.y - sf2D*.5;
    
    // Place a crater beneath the object.
    vec3 q2 = p - vec3(0, 1, 0) - vec3(0, 5. - .55 - 1., 0);
    ter = smax(ter, -(length(q2) - 5.), .5);
    ter += (.0 - sf2D*.5); 
    
    
 
    // Hollowing the tunnel out of the terrain. Not used here.
    //ter = smax(ter, tun, 3.);
    
    // The polyhedral object.
        
    // Face, line and vertex distances. 
    float face = 1e5, line = 1e5, vert = 1e5;

 
    
    #ifdef PENTAKIS_ICOSAHEDRON
    
        // A Pentakis icosahedron: Like an icosahedron, but with 80 sides.
    
        // Construct a regular 20 sided icosahedron, then use the vertices to 
        // subdivide into four extra triangles to produce an 80 sided Pentakis
        // icosahedron. Subdivision is achieved by using the known triangle 
        // face vertex points to precalculate the edge points and triangle center 
        // via basic trigonometry. See e0, e1, e2 above.
        //
        // On a side note, I'd imagine there's a way to fold space directly into a 
        // Pentakis icosahedron, but I got lazy and took the slower subdivided 
        // icosahedron route. If anyone knows how to do it more directly, feel free 
        // to let me know.


        // Local object cell coordinates.
        vec3 objP = opIcosahedron(q);

        // Vertices.
        vert = min(vert, length(objP - v0) - .05); 
        vert = min(vert, length(objP - e0) - .05); 

        // Lines or edges.
        line = min(line, sdCapsule(objP, v0, e0) - .02);
        line = min(line, sdCapsule(objP, e0, e2) - .02);

        float ndg = .97;

        // Vertex triangle facets -- Due to the nature of folding space,
        // all three of these are rendered simultaneously.
        face = min(face, udTriangle(objP, v0*ndg, e0*ndg, e2*ndg) - .03);
        // Middle face.
        face = min(face, udTriangle(objP, e0*ndg, e1*ndg, e2*ndg) - .03);
    
    #else
    
        // The second polyhedral object option:
        //
        // This is an exert from Knighty's awesome Polyhedras with control example, here:
        // https://www.shadertoy.com/view/MsKGzw
        //
        // Here's a very brief explanation: Folding space about various planes will produce various
        // objects -- The simplest object would be a cube, where you're folding once about the YZ, XZ, 
        // and XY planes: p = abs(p); p -= vec3(a, b, c); dist = max(max(p.x, p.y), p.z);
        //
        // Things like icosahedrons require more folds and more advanced plane calculations, but the
        // idea is the same. It's also possible to mix objects, which is what Knighty has cleverly 
        // and elegantly done here. In particular, a Triakis icosahedron (bas = vec3(1, 0, 0), an 
        // icosahedron (bas = vec3(0, 1, 0) and a dodecahedron (bas = vec3(0, 0, 1). Mixtures, like 
        // the default (bas = vec3(1), will give you a compounded mixture of three platonic solids, 
        // which each have their own names, but I'll leave you to investigate that. :)

        // Setup: Folding plane calculation, etc. I've made some minor changes, but it's essesntially
        // Knighty's original code.
        //
        // Animating through various platonic solid compounds.
        //vec3 bas = vec3(sin(iTime/4.)*.5 + .5, cos(iTime*1.25/4.)*.5 + .5,  cos(iTime/1.25/4.)*.5 + .5);
        // A nice blend of all three base solids.
        const vec3 bas = vec3(1);

        vec3 pbc = vec3(scospin, 0, .5); // No normalization so 'barycentric' coordinates work evenly.
        vec3 pca = vec3(0, scospin, cospin);
        //U, V and W are the 'barycentric' coordinates. Not sure if they are really barycentric...
        vec3 pg = normalize(mat3(pab, pbc, pca)*bas); 
        // For slightly better DE. In reality it's not necesary to apply normalization. :) 
        pbc = normalize(pbc); pca = normalize(pca);
        
        p = q; // Initial coordinates set to the ball's position.

        // Fold space.
        for(int i = 0; i<5; i++){
            p.xy = abs(p.xy); // Fold about xz and yz planes.
            p -= 2.*min(dot(p, nc), 0.)*nc; // Fold about nc plane.
        }

        // Analogous to moving local space out to the surface.
        p -= pg;

        // Object face distance.
        float d0 = dot(p, pab), d1 = dot(p, pbc), d2 = dot(p, pca);
        face = max(max(d0, d1), d2);
        //face -= abs((d0 + d1 + d2)/3. - face)*.25; // Subtle surface sparkle.

        // Object line distance.
        float dla = length(p - min(p.x, 0.)*vec3(1, 0, 0));
        float dlb = length(p - min(p.y, 0.)*vec3(0, 1, 0));
        float dlc = length(p - min(dot(p,nc), 0.)*nc);
        line = min(min(dla, dlb), dlc) - .025;

        // Vertices.
        vert = length(p) - .05;
    #endif
 
   
    
    // Storing the terrain, line, face and vertex information.
    vRID = vec4(ter, line, face, vert);

    // Return the minimum distance.
    return min(min(ter, line), min(face, vert));
 
}



// Basic raymarcher.
float trace(in vec3 ro, in vec3 rd){

    float t = 0., h;
    
    for(int i=0; i<96; i++){
    
        h = map(ro + rd*t);
        // Note the "t*b + a" addition. Basically, we're putting less emphasis on accuracy, as
        // "t" increases. It's a cheap trick that works in most situations... Not all, though.
        if(abs(h)<DELTA*(t*.1 + 1.) || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.), etc.
        
        // Ray shortening isn't really needed here, so call me paranoid. :D
        t += h*.9;
    }

    return min(t, FAR);
}

/*
// Tetrahedral normal - courtesy of IQ. I'm in saving mode, so the two "map" calls saved make
// a difference. Also because of the random nature of the scene, the tetrahedral normal has the 
// same aesthetic effect as the regular - but more expensive - one, so it's an easy decision.
vec3 normal(in vec3 p)
{  
    vec2 e = vec2(-1., 1.)*0.001;   
    return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
                     e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}
*/

 
// Standard normal function. It's not as fast as the tetrahedral calculation, but more symmetrical.
vec3 normal(in vec3 p, float ef) {
    vec2 e = vec2(0.001*ef, 0);
    return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
}

  
// Surface bump function..
float bumpSurf3D( in vec3 p){
    
    float n = 0.;
 
    
    // Rocks.
    if(svRID == 0){
        
        p *= 4.;
        n = surfFunc2D(p);
        //n = mix(n, 1.-surfFunc2D(p*4.), .25);
        //n = mix(n, surfFunc2D(p*12.), .1); 
    }
    else{
        
        // Sand.
        n = 1.;//sand(p.xz*1.25);
        //n = mix(.5, n, smoothstep(0., bordW, (bordCol0Col1)));
       
/*       
        // Sand pattern alternative.
        p *= vec3(1.65, 2.2, 3.85)/1.25;
        //float ns = n2D(p.xz)*.57 + n2D(p.xz*2.)*.28 + n2D(p.xz*4.)*.15;
        float ns = n3D(p)*.57 + n3D(p*2.)*.28 + n3D(p*4.)*.15;

        // vec2 q = rot2(-3.14159/5.)*p.xz;
        // float ns1 = grad(p.z*32., 0.);//*clamp(p.y*5., 0., 1.);//smoothstep(0., .1, p.y);//
        // float ns2 = grad(q.y*32., 0.);//*clamp(p.y*5., 0., 1.);//smoothstep(0., .1, p.y);//
        // ns = mix(ns1, ns2, ns);

        ns = (1. - abs(smoothstep(0., 1., ns) - .5)*2.);
        ns = mix(ns, smoothstep(0., 1., ns), .65);

        // Use the height to taper off the sand edges, before returning.
        //ns = ns*smoothstep(0., .2, p.y - .075);
    

        // A surprizingly simple and efficient hack to get rid of the super annoying Moire pattern 
        // formed in the distance. Simply lessen the value when it's further away. Most people would
        // figure this out pretty quickly, but it took far too long before it hit me. :)
        n = ns/(1. + gT*gT*.015);
*/        
        
        
    }
    
    
    
    
    //return mix(min(n*n*2., 1.), surfFunc3D(p*2.), .35);
    return n;//min(n*n*2., 1.);
    
    /*
    // Obtaining some terrain samples in order to produce a gradient
    // with which to distort the sand. Basically, it'll make it look
    // like the underlying terrain it effecting the sand. The downside
    // is the three extra taps per bump tap... Ouch. :) Actually, it's
    // not that bad, but I might attempt to come up with a better way.
    float n = txFace0(p);
    vec3 px = p + vec3(.001, 0, 0);
    float nx = txFace0(px);
    vec3 pz = p + vec3(0, 0, .001);
    float nz = txFace0(pz);
    
    // The wavy sand, that has been perturbed by the underlying terrain.
    return sand(p.xz + vec2(n - nx, n - nz)/.001*1.);
    */

}

// Standard function-based bump mapping routine: This is the cheaper four tap version. There's
// a six tap version (samples taken from either side of each axis), but this works well enough.
vec3 doBumpMap(in vec3 p, in vec3 nor, float bumpfactor){
    
    // Larger sample distances give a less defined bump, but can sometimes lessen the aliasing.
    const vec2 e = vec2(.001, 0); 
    
    // Gradient vector: vec3(df/dx, df/dy, df/dz);
    float ref = bumpSurf3D(p);
   
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx)) - ref)/e.x; 
    
    /*
    // Six tap version, for comparisson. No discernible visual difference, in a lot of cases.
    vec3 grad = vec3(bumpSurf3D(p - e.xyy) - bumpSurf3D(p + e.xyy),
                     bumpSurf3D(p - e.yxy) - bumpSurf3D(p + e.yxy),
                     bumpSurf3D(p - e.yyx) - bumpSurf3D(p + e.yyx))/e.x*.5;
    */ 
  
    // Adjusting the tangent vector so that it's perpendicular to the normal. It's some kind 
    // of orthogonal space fix using the Gram-Schmidt process, or something to that effect.
    grad -= nor*dot(nor, grad);          
         
    // Applying the gradient vector to the normal. Larger bump factors make things more bumpy.
    return normalize(nor + grad*bumpfactor);
    
}



// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float calcAO(in vec3 p, in vec3 n)
{
    float sca = 2., occ = 0.;
    for( int i = 0; i<5; i++ ){
    
        float hr = float(i + 1)*.2/5.;        
        float d = map(p + n*hr);
        occ += (hr - d)*sca;
        sca *= .7;
    }
    
    return clamp(1. - occ, 0., 1.);  
    
}

// Cheap shadows are hard. In fact, I'd almost say, shadowing particular scenes with limited 
// iterations is impossible... However, I'd be very grateful if someone could prove me wrong. :)
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 32; 
    
    ro += n*.0015;
    vec3 rd = lp - ro; // Unnormalized direction ray.
    

    float shade = 1.;
    float t = 0.;//.0015; // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), .0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i = min(iFrame, 0); i<maxIterationsShad; i++){

        float d = map(ro + rd*t);
        shade = min(shade, k*d/t);
        //shade = min(shade, smoothstep(0., 1., k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        t += clamp(d, .05, .5); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}

 


 
// Krzysztof Narkowicz's HDR color to LDR space using the ACES operator.
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve
//
// As mentioned on his blog, the following is roughly twice as strong as the 
// original, so needs to compensate. On a side note, Knarkowicz (user name) 
// has some awesome examples on Shadertoy, if you feel like looking them up. :)
 
vec3 ACESFilm(in vec3 x){
    // Different numbers to the original: They work here, but might not
    // be suitable for other examples. Having said that, these numbers
    // match more closely to Stephen Hill's original.
    float tA = .6275, tB = .015, tC = .6075, tD = .295, tE = .14; // x *= .5;
    //float tA = .9036, tB = .018, tC = .8748, tD = .354, tE = .14; // x *= .6;
    return clamp((x*(tA*x + tB))/(x*(tC*x + tD) + tE), 0., 1.);
}

/*

// I wrote a trimmed down single function version based on the following:
//
// Baking Lab by MJP and David Neubelt - http://mynameismjp.wordpress.com/
// All code licensed under the MIT license.
//
// Originally written by Stephen Hill (@self_shadow).

vec3 ACESFitted(vec3 col){
    
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT.
    const mat3 ACESIn = mat3(.59719, .35458, .04823, 
                             .07600, .90834, .01566, 
                             .02840, .13383, .83777);

    // ODT_SAT => XYZ => D60_2_D65 => sRGB.
    const mat3 ACESOut = mat3(1.60475, -.53108, -.07367, 
                              -.10208,  1.10813, -.00605, 
                              -.00327, -.07276,  1.07602);

    col *= ACESIn;

    // Apply RRT and ODT fit.
    col = (col*(col + .0245786) - .000090537)/(col*(.983729*col + .432951) + .238081);

    col *= ACESOut;

    // Range: [0, 1].
    return clamp(col, 0., 1.);
}

*/

// Cheap sky function, which includes coloring via scattering and some simple clouds.
// The cheap one sample atmospheric calculations are a reworking of Bearworks' 
// "Fast Atmospheric Scattering example, here: https://www.shadertoy.com/view/WtBXWw
//
vec3 getSky(vec3 ro, vec3 rd, vec3 ld){ 
    
    
    // Subtle, fake sky curvature: Defintiely not physically correct. In fact, you
    // can take it out, if it doesn't sit right with you.
    rd.z *= .95 + length(rd.xy)*.125;
    rd = normalize(rd);

    // The dot product of the incident ray and light (sun) direction, which is used
    // to position the sun. Dot(a, b) happens to equal the cosine of the angle between 
    // the two vectors, which is required for various scattering formulae.
    float sun = clamp(dot(rd, ld), 0., 1.);
    

    // Atmospheric scattering: Calculate the in-scattering and extinction, which
    // is essentially the light added and light removed, then combine them to 
    // produce the sky color. This is a very much simplified one sample version, but
    // it results in a blue looking sky, so it'll do.
    
    // In-scattering (Adds light) - Particles scattered into the line of sight.
    // This determines how much light is accumulated by scattering.

    // Out-scattering (Removes light) - Determines how much light is lost due to 
    // particles being scattered out of the line of sight.
    
    // Extinction: Total light removed via out-scattering or absorbtion.

    
    // The calculations focus on two different kinds of scattering, namely, Rayleight
    // and Mie scattering. Directly from Wikipedia, here are the definitions:
    
    // Rayleigh scattering describes the elastic scattering of light by spheres that 
    // are much smaller than the wavelength of light.
    vec3 Rayleigh = vec3(1);
    
    // Mie scattering occurs when the diameters of atmospheric particulates are similar 
    // to the wavelengths of the scattered light. Dust, pollen, smoke and microscopic 
    // water droplets that form clouds are common causes of Mie scattering.
    vec3 Mie = vec3(1); 
 

    // Radiance: Red, green and blue spectral distribution.
    //
    // Scattering coefficients: Derived from the refractive index of air.
    vec3 betaR = vec3(5.8e-2, 1.35e-1, 3.31e-1); // Rayleigh.
    vec3 betaM = vec3(4e-2); // Mie.
    // Other versions for a slightly different color.
    //vec3 betaR = vec3(6.95e-2, 1.18e-1, 2.44e-1); // Rayleigh.
    //vec3 betaM = vec3(4e-2);  // Mie.   
  
    // Optical depth -- Based on the zenith angle, which makes sense;
    // More particle density in the lower atmosphere, and all that.
    const float RayleighAtt = 1.;
    const float MieAtt = 1.2; 
    float zAng = max(1e-6, rd.y);

    // G is inversely proportional to the sun size.
    const float g = .95; // .75, etc.
    //
    // Haze particles and Mie scattering: Henyey-Greenstein phase function.
    Mie *= betaM/betaR/(4.*3.14159)*(1. - g*g)/pow(1. + g*g - 2.*g*sun, 1.5);

    // In-scatter - Klassen's model.
    vec3 inScatter = (Mie + Rayleigh)*(1. + sun*sun);
    
    // Light attenuation via extinction: More absorbtion occurs in the denser lower 
    // atmosphere, than the higher lighter one. If you look at the Rayleigh scattering 
    // coefficient, you'll see that more blue and green is taken out, so things appear 
    // redder when the sun appears in the lower atmosphere -- Like at the horizon, so 
    // no surprises there.
    //
    // Extinction, which is the combination of absorption and out scattering.
    vec3 extinction = exp(-(betaR*RayleighAtt + betaM*MieAtt)/zAng);
    
    // Produce the sky.
   
    // Sky: Combine the incoming light with the extincted light to produce the sky color.   
    vec3 col = inScatter*(1. - extinction);
          
    // Sun.
    col += vec3(1.6, 1.4, 1)*pow(sun, 350.)*extinction*.5;
    
    // Sun haze.
    col += vec3(.8, .9, 1)*pow(sun, 2.)*extinction*.4;
      
    //col *= vec3(1., 1.1, 1.2)*.85; // Vibrancy tweak, or cheat. 
    
  
    // I'm pretty sure I'm not the only one who doesn't enjoy color space conversion,
    // and the full gamut (pun intended) of less-than-helpful protracted contradictory 
    // articles that attempt to explain it all. :D
    //
    // Anyway, from what I understand, the calculations above are based on HDR figures, 
    // so we need to take them down to LDR (low dynamic range), or something... At the
    // end of the day, this function will effect that, so it needs to be done. In fact, 
    // it's a rough conversion, but it's good enough for me. If you'd like expert 
    // analysis on this super boring subject, feel free to consult the experts. :D
     
    // Narkowicz's clever, much smaller approximation... which I've modified to make it 
    // match the original conversion below.
    col = ACESFilm(col);
    // The original function: I guess it's more accurate, but looks virtually the same 
    // in this particular case, so I don't see the need to use the extra code.
    //col = ACESFitted(col);
 
 
    // A simple way to place some clouds on a distant plane above the terrain -- 
    // Based on something IQ uses.
    const float SC = 1e5;
    float t = (SC - ro.y - .15)/(rd.y + .15); // Trace out to a distant XZ plane.
    vec2 uv = (ro + t*rd).xz; // UV coordinates.
    
    // Mix the sky with the clouds, whilst fading out a little toward the horizon.
    if(t>0.) col = mix(col, vec3(2), smoothstep(.45, 1., fBm(2.*uv/SC))*
                       smoothstep(.45, .65, rd.y*.5 + .5)*.4);        
  
    // Return the clamped sky color.
    return clamp(col, 0., 1.);
    
} 


// Smooth fract function.
float sFract(float x, float sf){
    
    x = fract(x);
    return min(x, (1. - x)*x*sf);
    
}

// hash based 3d value noise
vec4 hash41(vec4 p){
    return fract(sin(p)*43758.5453);
}

// Compact, self-contained version of IQ's 3D value noise function.
float n3D(vec3 p){
    
    const vec3 s = vec3(27, 111, 57);
    vec3 ip = floor(p); p -= ip; 
    vec4 h = vec4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); 
    //p *= p*p*(p*(p*6. - 15.) + 10.);
    h = mix(hash41(h), hash41(h + s.x), p.x);
    h.xy = mix(h.xz, h.yw, p.y);
    return mix(h.x, h.y, p.z); // Range: [0, 1].
}

// The grungey texture -- Kind of modelled off of the metallic Shadertoy texture,
// but not really. Most of it was made up on the spot, so probably isn't worth 
// commenting. However, for the most part, is just a mixture of colors using 
// noise variables.
vec3 GrungeTex(in vec3 p){
    
    // Some fBm noise.
    //float c = n2D(p*4.)*.66 + n2D(p*8.)*.34;
    float c = n3D(p*3.)*.57 + n3D(p*7.)*.28 + n3D(p*15.)*.15;
   
    // Noisey bluish red color mix.
    vec3 col = mix(vec3(.25, .115, .02), vec3(.35, .5, .65), c);
    // Running slightly stretched fine noise over the top.
    col *= n3D(p*vec3(150., 150., 150.))*.5 + .5; 
    
    // Using a smooth fract formula to provide some splotchiness... Is that a word? :)
    col = mix(col, col*vec3(.75, .95, 1.1), sFract(c*4., 12.));
    col = mix(col, col*vec3(1.2, 1, .8)*.8, sFract(c*5. + .35, 12.)*.5);
    
    // More noise and fract tweaking.
    c = n3D(p*8. + .5)*.7 + n3D(p*18. + .5)*.3;
    c = c*.7 + sFract(c*5., 16.)*.3;
    col = mix(col*.6, col*1.4, c);
    
    // Clamping to a zero to one range.
    return clamp(col, 0., 1.);
    
}



void mainImage( out vec4 fragColor, in vec2 fragCoord ){    


    
    // Screen coordinates.
    vec2 u = (fragCoord - iResolution.xy*.5)/iResolution.y;
    
    // Camera Setup.     
    vec3 lookAt = vec3(-1, 1, 0); // Camera position, doubling as the ray origin.
    vec3 ro = lookAt + vec3(sin(iTime/4.), .0, -4);  // "Look At" position.
    
    //vec3 lp = vec3(0, 0, ro.z + 8.);
    // Usually, you'd just make this a unit directional light, and be done with it, but I
    // like some of the angular subtleties of point lights, so this is a point light a
    // long distance away. Fake, and probably not advisable, but no one will notice.
    vec3 lp = vec3(0, 0, ro.z) + vec3(-FARSUN*.35, FARSUN*(sin(iTime)*.15*0. + .15), FARSUN);
    
    // Using the Z-value to perturb the XY-plane.
    // Sending the camera and "look at" vectors down the tunnel. The "path" function is 
    // synchronized with the distance function.
    ro.xy += path(ro.z);
    lookAt.xy += path(lookAt.z);
    //lp.xy += path(lp.z);
 
    
    // Using the above to produce the unit ray-direction vector.
    float FOV = 3.14159265/3.; // FOV - Field of view.
    vec3 forward = normalize(lookAt - ro);
    vec3 right = normalize(vec3(forward.z, 0, -forward.x )); 
    vec3 up = cross(forward, right);

    // rd - Ray direction.
    vec3 rd = normalize(forward + FOV*u.x*right + FOV*u.y*up);
    
    // Swiveling the camera about the XY-plane (from left to right) when turning corners.
    // Naturally, it's synchronized with the path in some kind of way.
    rd.xy = rot2( path(lookAt.z).x/32.)*rd.xy;
    
    vec3 aCol = vec3(0);
    
   
    vec3 cam = ro;
    vec3 sp = ro; 
    
    float alpha = 1.;
    float gSh = 1.;
    float objRef = 1.;
    
    for(int j = 0; j<PASSES; j++){

        // Raymarching.
        float t = trace(sp, rd);

        gT = t;

        svVRID = vRID;
        svRID = vRID[0]<vRID[1] && vRID[0]<vRID[2] && vRID[0]<vRID[3]? 0 : 
                vRID[1]<vRID[2] && vRID[1]<vRID[3]? 1 : vRID[2]<vRID[3]? 2 : 3;


        // Sky. Only retrieving a single color this time.
        //vec3 sky = getSky(rd);

        // The passage color. Can't remember why I set it to sky. I'm sure I had my reasons.
        vec3 col = vec3(0);

 
        
        // Advance the ray to the surface. This becomes the new ray origin for the
        // next pass.
        sp += rd*t;

        //float pathHeight = sp.y;//surfFunc(sp);// - path(sp.z).y; // Path height line, of sorts.

        // If we've hit the ground, color it up.
        if (t < FAR){


             
            vec3 sn = normal(sp, 1.); // Surface normal. //*(1. + t*.125)
            
            
         
            // Light direction vector. From the sun to the surface point. We're not performing
            // light distance attenuation, since it'll probably have minimal effect.
            vec3 ld = lp - sp;
            float lDist = max(length(ld), 0.001);
            ld /= lDist; // Normalize the light direct vector.

            lDist /= FARSUN; // Scaling down the distance to something workable for calculations.
            float atten = 1./(1. + lDist*lDist*.025);


            // Texture scale factor.        
            const float tSize = 1./8.;

            // Extra shading in the sand crevices.
            float bSurf = bumpSurf3D(sp);

            vec3 oSn = sn;

            //float bf = svRID == 0? .5 : .05;
            sn = doBumpMap(sp, sn, .1/(1. + t*t*.001));

  
            // The reflective ray, which tends to be very helpful when
            // calculating reflections. :)
            vec3 reflection = reflect(rd, sn);

            // Soft shadows and occlusion.
            float sh = softShadow(sp, lp, sn, 8.); 
            float ao = calcAO(sp, sn); // Amb, 6.ient occlusion.


            float dif = max( dot( ld, sn ), 0.); // Diffuse term.
            float spe = pow(max( dot( reflect(-ld, sn), -rd ), 0.), 32.); // Specular term.
            float fre = clamp(1.0 + dot(rd, sn), 0., 1.); // Fresnel reflection term.

            // Schlick approximation. I use it to tone down the specular term. It's pretty subtle,
            // so could almost be aproximated by a constant, but I prefer it. Here, it's being
            // used to give a sandstone consistency... It "kind of" works.
            float Schlick = pow( 1. - max(dot(rd, normalize(rd + ld)), 0.), 5.);
            float fre2 = mix(.2, 1., Schlick);  //F0 = .2 - Dirt... or close enough.

            // Overal global ambience. It's made up, but I figured a little occlusion (less ambient light
            // in the corners, etc) and reflectance would be in amongst it... Sounds good, anyway. :)
            float amb = ao*.25;

 
            // 2D surface function.
            float sf2D = surfFunc2D(sp);

             // Terrain.
            if(svRID==0){

                // Coloring the soil.
                col = min(vec3(1.2, .75, .5)*vec3(1, .9, .8), 1.);
                
                //vec3 colR = mix(vec3(1, .8, .5), vec3(.5, .25, .125), clamp((sp.y*1.5 + .5), 0., 1.));
                //col = mix(col, colR, .5);
                
                col *= mix(vec3(1.5, 1, .8), 1./vec3(1.2, 1, .8), sf2D);
                col = mix(col/vec3(1.1, 1, .9), col*vec3(1.1, 1, .9), abs(sn));
                objRef = .0;

            } 

            // Polyhedral lines.
            if(svRID==1){
                col = vec3(.2);
                objRef = .25;
                
            } 

            // Polyhedral mirror faces.
            if(svRID==2){
                col = vec3(1);//vec3(1, .7, .3)*4.;
                objRef = 1.;
                
            }           

            // Polyhedral vertices.
            if(svRID==3){
                col = vec3(.1);
                objRef = 1.;
                
            }
            
            // Finer details.
            //col = mix(col/1.35, col*1.35, bSurf);

            // Grungey overlay: Add more to the rock surface than the sand.
            // Surface texel.
            vec3 txP = sp;
            if(svRID>0) txP = moveBall(txP);//rot3(txP - vec3(0, 1.25/2., 0), vec3(3.14159/6.), iTime/2.);
            vec3 tx = GrungeTex(txP/4.);//*vec3(1.2, 1.15, 1.05);//
            col = mix(col, col*tx*3., .5); 

            


            // Combining all the terms from above. Some diffuse, some specular - both of which are
            // shadowed and occluded - plus some global ambience. Not entirely correct, but it's
            // good enough for the purposes of this demonstation.        
            col = col*(dif + amb + vec3(1, .97, .92)*spe*fre2*1.);


 
            
            // A bit of sky reflection. Not really accurate, but I've been using fake physics since the 90s. :)
            vec3 refSky = getSky(sp, reflect(rd, sn), ld);
            
       
            // Applying the shadows and ambient occlusion.
            col = col*ao*atten*(sh + .25);
            
            // Sky reflection on the terrain. Considering this is a multiple reflection, I could
            // literaaly reflect the sky. However, setting the terrain to no reflections and adding
            // this looks roughly the same and is way, way cheaper.
            if(svRID==0){
                col += refSky*ao*atten*(sh + .75)*.05;
            }
            
            
            // Set the unit direction ray to the new reflected direction, and bump the 
            // ray off of the hit point by a fraction of the normal distance. Anyone who's
            // been doing this for a while knows that you need to do this to stop self
            // intersection with the current launch surface from occurring... It used to 
            // bring me unstuck all the time. I'd spend hours trying to figure out why my
            // reflections weren't working. :)
            rd = reflection;
            sp += sn*DELTA*1.1;

        }


        // Combine the scene with the sky.
        vec3 sky = getSky(ro, rd,  normalize(lp - sp));
        //sky = mix(sky, 1. - exp(-sky), .15);
        //col = mix(col, sky, min(t*t*1.5/FAR/FAR, 1.)); // Quadratic fade off. More subtle.
        col = mix(min(col, 1.), sky, smoothstep(0., .99, t/FAR)); // Linear fade. Much dustier. I kind of like it.
        
        // Add the layer color to the overall total.
        aCol += min(col, 1.)*alpha; 
        
        // If the hit object's reflective factor is zero, or the ray has reached
        // the far horizon, break.
        if(objRef < .001 || t >= FAR) break;
        
        // Decrease the alpha factor (ray power of sorts) by the hit object's reflective factor.
        alpha *= objRef;

    } 
    
    // Greyish tone.
    //col = mix(col, vec3(1)*dot(col, vec3(.299, .587, .114)), .5);
    
    
    // Standard way to do a square vignette.
    u = fragCoord/iResolution.xy;
    aCol = min(aCol, 1.)*pow( 16.*u.x*u.y*(1. - u.x)*(1. - u.y) , .0625);
 
  
    // Done.
    fragColor = vec4(sqrt(clamp(aCol, 0., 1.)), 1);
}












// https://www.shadertoy.com/view/WdtBzn

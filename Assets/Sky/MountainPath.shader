
Shader "Skybox/MountainPath"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
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

#define mod(x,y) (x-y*floor(x/y)) // glsl mod
#define iTime _Time.y
#define iResolution _ScreenParams
#define vec2 float2
#define vec3 float3
#define vec4 float4
#define mix lerp
#define texture tex2D
#define fract frac
#define mat4 float4x4
#define mat3 float3x3
#define textureLod(a,b,c) tex2Dlod(a,float4(b,0,c))
/*

    Mountain Path
    -------------

    A path running through some dry rocky terrain. Rendered in a pseudo low-poly style.

    This was orginally supposed to be a simple curvy path through a field, but then I wondered what
    stepping the path would look like, then I wanted to see what it'd look like going through
    something... so now, I'm not really sure what it is, or why anyone in their right mind would 
    build a ridiculously long path in the middle of a mountainous pass that leads to nowhere. :)

    I'm fond of the old demo scenes and those pictures on 80s sci-fi novels, so I went for a psuedo 
    low-poly look with a little realism thrown into the lerp. The geometry was surprisingly simple to 
    produce - I thought the stairs might present some problems, but it just involved stepping the 
    path heights and dealing with the overlap.

    The low poly terrain is just a sinusoidal-based layer with some relatively cheap cellular noise 
    added to it. Anyway, this is just a practice run for a more interesting variation I have in mind.
    


*/

#define FAR 80. // Maximum ray distance. Analogous to the far plane.


// Scene object ID. Either the path (0) or the surroundings (1).
float objID;
float svObjID; // Global ID to keep a copy of the above from pass to pass.

// Fabrice's concise, 2D rotation formula.
float2x2 r2(float th){ float2 a = sin(float2(1.5707963, 0) + th); return float2x2(a, -a.y, a.x); }

// float3 to float hash.
float hash31( float3 p ){ return frac(cos(dot(p, float3(157, 113, 7)))*45758.5453); }

// Tri-Planar blending function. Based on an old Nvidia tutorial.
float3 tex3D( sampler2D t, in float3 p, in float3 n ){
    
    n = max(abs(n), 0.001);
    n /= dot(n, float3(1,1,1));
    float3 tx = tex2D(t, p.yz).xyz;
    float3 ty = tex2D(t, p.zx).xyz;
    float3 tz = tex2D(t, p.xy).xyz;
    
    // tex2Ds are stored in sRGB (I think), so you have to convert them to linear space 
    // (squaring is a rough approximation) prior to working with them... or something like that. :)
    // Once the final color value is gamma corrected, you should see correct looking colors.
    return (tx*tx*n.x + ty*ty*n.y + tz*tz*n.z);
}


float drawObject(in float3 p){
    
    // Wrap conditions:
    // Anything that wraps the domain will work.
    //p = cos(p*6.2831853)*.25 + .25; 
    //p = abs(cos(p*3.14159)*.5);
    //p = frac(p) - .5; 
    //p = abs(frac(p) - .5); 
    
    // Distance metrics:
    // Here are just a few variations. There are way too many to list them all,
    // but you can try combinations with "min," and so forth, to create some
    // interesting combinations.
    
    // Spherical. (Square root needs to be factored to "d" in the cellTile function.)
    //p = frac(p) - .5;    
    //return dot(p, p)/1.5;
    
    // Octahedral... kind of.
    //p = abs(frac(p)-.5);
    //return dot(p, float3(.333));
    
    // Triangular.
    //p = frac(p) - .5;
    //p = max(abs(p)*.866025 + p.yzx*.5, -p);
    //return max(max(p.x, p.y), p.z);  

    
    // Cubic.
    p = abs(frac(p) - .5); 
    return max(max(p.x, p.y), p.z);
    
    // Cylindrical. (Square root needs to be factored to "d" in the cellTile function.)
    //p = frac(p) - .5; 
    //return max(max(dot(p.xy, p.xy), dot(p.yz, p.yz)), dot(p.xz, p.xz));
    
    // Octahedral.
    //p = abs(frac(p) - .5); 
    //p += p.yzx;
    //return max(max(p.x, p.y), p.z)*.5;

    // Hexagonal tube.
    //p = abs(frac(p) - .5); 
    //p = max(p*.866025 + p.yzx*.5, p.yzx);
    //return max(max(p.x, p.y), p.z);
    
    
}


// Repeat cellular tile routine. The operation count is extremely low when compared to conventional
// methods. No loops, no flooring, no hash calls, etc. Conceptually speaking, this is the fastest way 
// to produce a reasonable 3D cellular pattern... Although, there's one with three objects and no 
// rotation, but quality really suffers at that point. 
float cellTile(in float3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    float4 d; 
    d.x = drawObject(p - float3(.81, .62, .53));
    p.xy = float2(p.y - p.x, p.y + p.x)*.7071;
    //p.xy = float2(p.y*.866025 - p.x*.5, p.y*.5 + p.x*.866025); // Etc.
    d.y = drawObject(p - float3(.39, .2, .11));
    p.yz = float2(p.z - p.y, p.z + p.y)*.7071;
    //p.yz = float2(p.z*.866025 - p.y*.5, p.z*.5 + p.y*.866025); // Etc.
    d.z = drawObject(p - float3(.62, .24, .06));
    p.xz = float2(p.z - p.x, p.z + p.x)*.7071;
    //p.xz = float2(p.z*.866025 - p.x*.5, p.z*.5 + p.x*.866025); // Etc.
    d.w = drawObject(p - float3(.2, .82, .64));

    // Obtain the minimum, and you're done.
    d.xy = min(d.xz, d.yw);
        
    return 1. - min(d.x, d.y)*2.; // Scale between zero and one... roughly.
    
    // For anyone wanting to experiment with this, the following gives better variance:
    //const float scale = 1.; // 1 up to 4, or higher, depending on the look you want.
    // Obviously, for the reverse, you take the one and minus away.
    //return 1. - min(min(d.x, d.y)*2.*scale, 1.);
    
}

/*
// Second order version.
float cellTile(in float3 p){
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    float4 v, d; 
    d.x = drawObject(p - float3(.81, .62, .53));
    p.xy = float2(p.y - p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - float3(.39, .2, .11));
    p.yz = float2(p.z - p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - float3(.62, .24, .06));
    p.xz = float2(p.z - p.x, p.z + p.x)*.7071;
    d.w = drawObject(p - float3(.2, .82, .64));

    v.xy = min(d.xz, d.yw), v.z = min(max(d.x, d.y), max(d.z, d.w)), v.w = max(v.x, v.y); 
   
    d.x =  min(v.z, v.w) - min(v.x, v.y); // First minus second order, for that beveled Voronoi look. Range [0, 1].
    //d.x =  min(v.x, v.y); // Minimum, for the cellular look.
        
    return d.x*2.; // Normalize.
    
}
*/



// The path is a 2D sinusoid that varies over time, depending upon the frequencies, and amplitudes.
float2 path(in float z){ 

    //return float2(0); // Path 1.
    //return float2(sin(z*.05)*cos(z*.1)*2.5, sin(z*.06)*2.); // Path 2.
    return float2(sin(z*.15)*2.5, cos(z*.1)); // Path 3.
}

// Stair path.
float2 sPathF(in float2 p){ 

    //return float2(0); // Path 1.
    //return sin(p*.06)*2.; // Path 2.
    return cos(p*.1); // Path 3.
}

// The triangle function that Shadertoy user Nimitz has used in various triangle noise demonstrations.
// See Xyptonjtroz - Very cool. Anyway, it's not really being used to its full potential here.
//float3 tri(in float3 x){return abs(x-floor(x)-.5);} // Triangle function.

// The function used to perturb the walls of the cavern: There are infinite possibities, but this one is 
// just a cheap...ish routine - based on the triangle function - to give a subtle jaggedness. Not very fancy, 
// but it does a surprizingly good job at laying the foundations for a sharpish rock face. Obviously, more 
// layers would be more convincing. However, this is a GPU-draining distance function, so the finer details 
// are bump mapped.
float surfFunc(in float3 p){
    
    return cellTile(p/8.);//*.8 + dot(tri(p*0.384*2. + tri(p.yzx*0.192*2.)), float3(0.666))*.2;
     
    // More interesting formations, and still quick, but not fast enough for this example.
    //return cellTile(p/10.)*.75 + cellTile(p/10.*3.)*.25; 
 
    // Very cheap triangle noise. Looks OK, all things considering.
    //p /= 2.5;
    //float n = dot(tri(p*0.5 + tri(p.yzx*0.25)), float3(0.666));
    //return n*.75 + dot(tri(p*0.75 + tri(p.yzx*0.375)), float3(0.666))*.25;

}



// Smooth maximum, based on IQ's smooth minimum.
float smax(float a, float b, float s){
    
    float h = clamp(.5 + .5*(a - b)/s, 0., 1.);
    return lerp(b, a, h) + h*(1. - h)*s;
}

// Drawing repeat objects right up again one another causes inaccuracies, so you get 
// around that by rending two sets of repeat objects in each dimension. Two for one
// axis, four for two axes, and eight (I think) for three.
//
// Basically, the aforementioned is just a way to say that to draw stairs along the 
// Z-axis, you need to draw two sets of repeat boxes. The boxes are aligned with the 
// path position. In the case of the step portion, the path's height has to be 
// snapped to a quantized number. You do that via flooring, etc.
float stairs(in float3 p, float2 pth){
   
    const float sc = 2.; // Stair scaling factor. It affects the length.
    // The quantized stair heights. Basically, making the surface flat. Two heights are 
    // being passed in to account for the two boxes we have to render to account for the
    // overlap.
    float2 iPthY = sPathF(floor(float2(p.z/sc, p.z/sc + .5))*sc);
    // Snapping the stair height to factors of four. Makes the step layers equal height.
    iPthY = floor(iPthY*4.)/4. - 2.5;   

    // Railings. Draw one railing using the path's X and Y positions, then use the
    // "abs" repeat trick to render the other one at the same time.
    float sY = abs(p.y - pth.y + 2.); // Railing height.
    p.x = abs(p.x - pth.x); // Railing X-position.

    // Railing, with a bit carved out.
    float rails = max(abs(p.x - 1.75 + .35/2.) - .35/2., sY - .85);
    float rails2 = max(abs(p.x - 1.75 + .35/2. + .3) - .35/2., sY - .65);
    rails = max(rails, -rails2);
    
    // Stair render.
    p.z /= sc;
    
    float2 iy = p.yy - iPthY; // Quantized stair heights.
    // Render a couple of boxes, then take the minimum.
    float2 qz  = abs(frac(float2(p.z, p.z + .5)) - .5); 
    float2 n = max(max(p.xx - 1.7, qz - .27), abs(iy) - .75);

    // Return the path object - the minimum of the stairs and the railings.
    return min(rails, min(n.x, n.y));
     
    
}

// The refracion distance field. It's exactly the same as above, but doesn't include
// the water plane. It's here to save cycles.
float map(float3 p){
    
    float2 pth = path(p.z);
    
    float sf = surfFunc(p); // Surface perturbation.

    // The terrain base layer.
    float ter = p.y - 3. + dot(sin(p*3.14159/18. - cos(p.yzx*3.14159/18.)), float3(3,3,3)); // 6. smoothing factor.
    //float ter = p.y - 4. + dot(sin(p*3.14159/16.), cos(p.yzx*3.14159/32.))*3.; // 4. smoothing factor.

    float st = stairs(p, pth); // The physical path. Not to be confused with the camera path.

    p.xy -= pth; // Wrap the tunnel around the path.

    float n = 1.5 - length(p.xy*float2(.5, 1)); // The tunnel to bore through the rock.
    n = smax(n + (.5 - sf)*1.5, ter + (.5 - sf)*3., 6.); // Smoothly boring the tunnel through the terrain.
    n = smax(n, -max(abs(p.x) - 1.75, abs(p.y + 1.5) - 1.5), .5); // Clearing away the rock around the stairs.
 
    // Object ID.
    objID = step(n, st); // Either the physical path or the surrounds.
    
    return min(n, st)*.866; // Return the minimum hit point.
 
}
 

// Pavers. Standard grid stuff.
float paver(float2 p, float mortW){
    
    
    float2 q = abs(frac(p + float2(.5, .5)) - .5);
    
    float d = max(q.x, q.y) - .5;
    
    //float c = smoothstep(0., mortW, min(q.x, q.y));
    float c = smoothstep(0., .02, abs(d) - mortW/3.);
    //if (q.x<.05 || q.y<.05) c *= .5;

    return c;

    
}


// Surface bump function. Tiles are fiddly, but simple enough. Basically, the surface
// normal is used to determine the 2D plane we wish to tile, then it's passed to the
// tile function.
float tiles( in float3 p, in float3 n,  float mortW){
    
    p.xy -= path(p.z);

    n = abs(n);
    
    float c = 1.;
    
    if (n.x>0.5) {
        
        if(p.y<-1.35) return 1.;
        p.xy = p.yz;

    }
    else if (n.y>0.5) {
         
       if(p.y>-1.35) p.x += sign(p.x)*.25;
        
        p.xy = p.xz;
        
    }
    
    return paver(p.xz, mortW);
    
}

// The bump function.
float bumpFunc(float3 p, float3 n){


    float c;
    if(svObjID>.5 ) c = 1. - surfFunc(p*3.); // cellTile(p/8.*3.);
    else c = tiles(p.xyz, n, .1);
    
    // Note that I could perform two returns and dispense with the float declaration,
    // but some graphics cards used to complain. I think all of them should be
    // fine now, but just in case.
    return c; 

}

// Standard function-based bump mapping function with some edging thrown into the lerp.
float3 doBumpMap(in float3 p, in float3 n, float bumpfactor, inout float edge, inout float crv){
    
    // Resolution independent sample distance... Basically, I want the lines to be about
    // the same pixel with, regardless of resolution... Coding is annoying sometimes. :)
    float2 e = float2(8./450., 0);   //float2(8./iResolution.y, 0); 
    
    float f = bumpFunc(p, n); // Hit point function sample.
    
    float fx = bumpFunc(p - e.xyy, n); // Nearby sample in the X-direction.
    float fy = bumpFunc(p - e.yxy, n); // Nearby sample in the Y-direction.
    float fz = bumpFunc(p - e.yyx, n); // Nearby sample in the Y-direction.
    
    float fx2 = bumpFunc(p + e.xyy, n); // Sample in the opposite X-direction.
    float fy2 = bumpFunc(p + e.yxy, n); // Sample in the opposite Y-direction.
    float fz2 = bumpFunc(p + e.yyx, n);  // Sample in the opposite Z-direction.
    
     
    // The gradient floattor. Making use of the extra samples to obtain a more locally
    // accurate value. It has a bit of a smoothing effect, which is a bonus.
    float3 grad = float3(fx - fx2, fy - fy2, fz - fz2)/(e.x*2.);  
    //float3 grad = (float3(fx, fy, fz ) - f)/e.x;  // Without the extra samples.


    // Using the above samples to obtain an edge value. In essence, you're taking some
    // surrounding samples and determining how much they differ from the hit point
    // sample. It's really no different in concept to 2D edging.
    edge = abs(fx + fy + fz + fx2 + fy2 + fz2 - 6.*f);
    edge = smoothstep(0., 1., edge/e.x*2.);
    
    
    // We may as well use the six measurements to obtain a rough curvature value while we're at it.
    //crv = clamp((fx + fy + fz + fx2 + fy2 + fz2 - 6.*f)*32. + .6, 0., 1.);
    
    // Some kind of gradient correction. I'm getting so old that I've forgotten why you
    // do this. It's a simple reason, and a necessary one. I remember that much. :D
    grad -= n*dot(n, grad);          
                      
    return normalize(n + grad*bumpfactor); // Bump the normal with the gradient floattor.
    
}

// tex2D bump mapping. Four tri-planar lookups, or 12 tex2D lookups in total. I tried to 
// make it as concise as possible. Whether that translates to speed, or not, I couldn't say.
float3 doBumpMap( sampler2D tx, in float3 p, in float3 n, float bf){
   
    const float2 e = float2(0.001, 0);
    
    // Three gradient floattors rolled into a matrix, constructed with offset greyscale tex2D values.    
    float3x3 m = float3x3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));
    
    float3 g = mul(float3(0.299, 0.587, 0.114),m); // Converting to greyscale.
    g = (g - dot(tex3D(tx,  p , n), float3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}



// Standard raymarching routine.
float trace(float3 ro, float3 rd){
   
    float t = 0., d;
    
    for (int i=0; i<160; i++){

        d = map(ro + rd*t);
        
        if(abs(d)<.001*(t*.125 + 1.) || t>FAR) break;
        
        t += d;
    }
    
    return min(t, FAR);
}


// Cheap shadows are the bain of my raymarching existence, since trying to alleviate artifacts is an excercise in
// futility. In fact, I'd almost say, shadowing - in a setting like this - with limited  iterations is impossible... 
// However, I'd be very grateful if someone could prove me wrong. :)
float softShadow(float3 ro, float3 lp, float k, float t){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 48; 
    
    float3 rd = lp-ro; // Unnormalized direction ray.

    float shade = 1.;
    float dist = .0025*(t*.125 + 1.);  // Coincides with the hit condition in the "trace" function.  
    float end = max(length(rd), 0.0001);
    //float stepDist = end/float(maxIterationsShad);
    rd /= end;

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){

        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.
        // So many options here, and none are perfect: dist += min(h, .2), dist += clamp(h, .01, stepDist), etc.
        dist += clamp(h, .07, .5); 
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (h<0.0 || dist > end) break; 
    }

    // I've added a constant to the final shade value, which lightens the shadow a bit. It's a preference thing. 
    // Really dark shadows look too brutal to me. Sometimes, I'll add AO also just for kicks. :)
    return min(max(shade, 0.) + .15, 1.); 
}


// Standard normal function. It's not as fast as the tetrahedral calculation, but more symmetrical. Due to 
// the intricacies of this particular scene, it's kind of needed to reduce jagged effects.
float3 getNormal(in float3 p) {
    const float2 e = float2(0.0025, 0);
    return normalize(float3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
}


/*
// Tetrahedral normal, to save a couple of "map" calls. Courtesy of IQ.
float3 getNormal( in float3 p ){

    // Note the slightly increased sampling distance, to alleviate
    // artifacts due to hit point inaccuracies.
    float2 e = float2(0.0025, -0.0025); 
    return normalize(
        e.xyy * map(p + e.xyy) + 
        e.yyx * map(p + e.yyx) + 
        e.yxy * map(p + e.yxy) + 
        e.xxx * map(p + e.xxx));
}
*/

// Normal calculation, with some edging and curvature bundled in.
float3 getNormal(float3 p, inout float edge, inout float crv) { 
    
    // Roughly two pixel edge spread, regardless of resolution.
    float2 e = float2(6./450., 0); //float2(6./iResolution.y, 0);

    float d1 = map(p + e.xyy), d2 = map(p - e.xyy);
    float d3 = map(p + e.yxy), d4 = map(p - e.yxy);
    float d5 = map(p + e.yyx), d6 = map(p - e.yyx);
    float d = map(p)*2.;

    edge = abs(d1 + d2 - d) + abs(d3 + d4 - d) + abs(d5 + d6 - d);
    //edge = abs(d1 + d2 + d3 + d4 + d5 + d6 - d*3.);
    edge = smoothstep(0., 1., sqrt(edge/e.x*2.));
/*    
    // Wider sample spread for the curvature.
    e = float2(12./450., 0);
    d1 = map(p + e.xyy), d2 = map(p - e.xyy);
    d3 = map(p + e.yxy), d4 = map(p - e.yxy);
    d5 = map(p + e.yyx), d6 = map(p - e.yyx);
    crv = clamp((d1 + d2 + d3 + d4 + d5 + d6 - d*3.)*32. + .5, 0., 1.);
*/
    
    e = float2(.0015, 0); //iResolution.y - Depending how you want different resolutions to look.
    d1 = map(p + e.xyy), d2 = map(p - e.xyy);
    d3 = map(p + e.yxy), d4 = map(p - e.yxy);
    d5 = map(p + e.yyx), d6 = map(p - e.yyx);
    
    return normalize(float3(d1 - d2, d3 - d4, d5 - d6));
}

// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float calcAO(in float3 pos, in float3 nor)
{
    float sca = 2.0, occ = 0.0;
    for( int i=0; i<5; i++ ){
    
        float hr = 0.01 + float(i)*0.5/4.0;        
        float dd = map(nor * hr + pos);
        occ += (hr - dd)*sca;
        sca *= 0.7;
    }
    return clamp( 1.0 - occ, 0.0, 1.0 );    
}

/*
// Simple environment mapping. Pass the reflected floattor in and create some
// colored noise with it. The normal is redundant here, but it can be used
// to pass into a 3D tex2D mapping function to produce some interesting
// environmental reflections.
//
// More sophisticated environment mapping:
// UI easy to integrate - XT95    
// https://www.shadertoy.com/view/ldKSDm
float3 eMap(float3 rd, float3 sn){

    float3 tx = tex3D(_MainTex, rd, sn);
    return smoothstep(.15, .5, tx); 
    
}
*/

// Cheap and nasty 2D smooth noise function, based on IQ's original. Very trimmed down. In fact,
// I probably went a little overboard. I think it might also degrade with large time values. I'll 
// swap it for something more robust later.
float n2D(float2 p) {

    float2 f = frac(p); p -= f; f *= f*(3. - f*2.);  
    
    return dot(mul(float2x2(frac(sin(float4(0, 41, 289, 330) + dot(p, float2(41, 289)))*43758.5453)),
                float2(1. - f.y, f.y)), float2(1. - f.x, f.x) );

}


// Simple fBm to produce some clouds.
float fbm(in float2 p){
    
    // Four layers of 3D noise.
    return 0.5333*n2D( p ) + 0.2667*n2D( p*2.02 ) + 0.1333*n2D( p*4.03 ) + 0.0667*n2D( p*8.03 );

}


// Pretty standard way to make a sky. 
float3 getSky(in float3 ro, in float3 rd, float3 lp){

    
    float sun = max(dot(rd, normalize(lp - ro)), 0.0); // Sun strength.
    float horiz = pow(1.0-max(rd.y, 0.0), 3.)*.35; // Horizon strength.
    
    //float3 col = lerp(float3(.6, .9, 1).zyx, float3(.62, .68, 1).zyx, rd.y*.5 + .5)*1.25*.5;
    
    // The blueish sky color. Tinging the sky redish around the sun.        
    float3 col = lerp(float3(.25, .35, .5), float3(.4, .375, .35), sun*.75);//.zyx;
    // lerping in the sun color near the horizon.
    col = lerp(col, float3(1, .9, .7), horiz);
    
    
    // Sun. I can thank IQ for this tidbit. Producing the sun with three
    // layers, rather than just the one. Much better.
    col += 0.25*float3(1, .7, .4)*pow(sun, 5.0);
    col += 0.25*float3(1, .8, .6)*pow(sun, 64.0);
    col += 0.15*float3(1, .9, .7)*max(pow(sun, 512.0), .25);
    
    // Add a touch of speckle. For better or worse, I find it breaks the smooth gradient up a little.
    col = clamp(col + hash31(rd)*0.04 - 0.02, 0., 1.);
    
    //return col;
    
    // Clouds. Render some 3D clouds far off in the distance. I've made them sparse and wispy,
    // since we're in the desert, and all that.
    
    // Mapping some 2D clouds to a plane to save some calculations. Raytrace to a plane above, which
    // is pretty simple, but it's good to have Dave's, IQ's, etc, code to refer to as backup.
    
    // Give the direction ray a bit of concavity for some fake global curvature - My own dodgy addition. :)
    //rd = normalize(float3(rd.xy, sqrt(rd.z*rd.z + dot(rd.xy, rd.xy)*.1) ));
    
    float t = (5000. - ro.y)/rd.y; // Trace out to a distant XZ plane.
    float2 uv = (ro + t*rd).xz; // UV coordinates.
    //float3 sc = float3(uv.x, 0., uv.y);
    
    // lerp the sky with the clouds, whilst fading out a little toward the horizon (The rd.y bit).
    if(t>0.) col = lerp( col, float3(1, .9, .8), 0.35*smoothstep(0.4, 1.0, fbm(.0005*uv)* clamp(rd.y*5., 0., 1.)));
    
    return col;

}

 

// Coloring\texturing the scene objects, according to the object IDs.
float3 getObjectColor(float3 p, float3 n){
    
    //p.xy -= path(p.z);
    // Object tex2D color, with some contract thrown in.
    float3 tx;
    //tx = smoothstep(.05, .5, tx);
    
    // Coloring the tunnel walls.
    if(svObjID>.5) {
        tx = tex3D(_MainTex, p/2., n );
        tx = smoothstep(-.1, .5, tx);
        tx *= float3(1, .6, .35); // Brownish.

        // Optional: Extra crevice darkening from biological buildup. Adds
        // depth - addition to the shadows and AO. 
        tx *= smoothstep(.1, .6, surfFunc(p))*.6 + .4;
        
        // Alternative algae in the crevices.
        //float c = smoothstep(.1, .6, surfFunc(p));
        //tx *= float3(c*c, c, c*c*c)*.6 + .4;
    }
    else {
        float2 pth = path(p.z);
        tx = tex3D(_MainTex, (p - float3(pth.xy - .5, .0)), n );
        tx = smoothstep(-.15, .5, tx);
        tx *= float3(1.5, 1.0, .5)*.65 + .5; // Tinting the stairs.
        tx *= tiles(p.xyz, n, .05)*.93 + .07;
        
        if(p.y - pth.y>-1.35) tx *= float3(.64, .62, .6); // Rails.
        
        
    }
    
    //tx *= bumpFunc(p.xyz, n);

    
    return tx;//pow(tx, float3(1.33))*1.66;
    
}

// Using the hit point, unit direction ray, etc, to color the scene. Diffuse, specular, falloff, etc. 
// It's all pretty standard stuff.
float3 doColor(in float3 ro, in float3 rd, in float3 lp, float t){
    
    // Initiate the scene (for this pass) to zero.
    float3 sceneCol = float3(0,0,0);
    
    if(t<FAR){ // If we've hit a scene object, light it up.
        
        // Surface hit point.
        float3 sp = ro + rd*t;

        // Retrieving the normal at the hit point, plus the edge and curvature values.
        float edge = 0., crv = 1.;
        float3 sn = getNormal(sp, edge, crv);


        float bf = .5;
        if(svObjID<.5) bf = .01;

        float edge2 = 0., crv2 = 1.; 
        //if(svObjID>.5)
        sn = doBumpMap(sp, sn, bf/(1. + t/FAR*.125), edge2, crv2); 

        bf = .07;
        
        float txF = 1.;
        if(svObjID<.5) {
            bf = .04;
            txF = 2.;
        }
        sn = doBumpMap(_MainTex, sp*txF, sn, bf);
        
        // Shading. Shadows, ambient occlusion, etc. We're only performing this on the 
        // first pass. Not accurate, but faster, and in most cases, not that noticeable.
        // In fact, the shadows almost didn't make the cut, but it didn't quite feel 
        // right without them.
        float sh = softShadow(sp + sn*.002, lp, 16., t); // Set to "1.," if you can do without them.
        float ao = calcAO(sp, sn);
        sh = (sh + ao*.3)*ao;
    
    
        float3 ld = lp - sp; // Light direction floattor.
        float lDist = max(length(ld), 0.001); // Light to surface distance.
        ld /= lDist; // Normalizing the light floattor.

        // Attenuating the light, based on distance.
        float atten = 3./(1. + lDist*0.01 + lDist*lDist*0.00008);

        // Standard diffuse term.
        float diff = max(dot(sn, ld), 0.);
        //diff = pow(diff, 2.)*.66 + pow(diff, 4.)*.34;
        // Standard specualr term.
        float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 32.0);
        //float fres = clamp(1. + dot(rd, sn), 0., 1.);
        //float Schlick = pow( 1. - max(dot(rd, normalize(rd + ld)), 0.), 5.0);
        //float fre2 = lerp(.5, 1., Schlick);  //F0 = .5.

        // Coloring the object. You could set it to a single color, to
        // make things simpler, if you wanted.
        float3 objCol = getObjectColor(sp, sn);

        // Combining the above terms to produce the final scene color.
        sceneCol = objCol*(diff + ao*.5 + float3(1, .7, .5)*spec*1.);
        
        // Edges.
        if(svObjID>.5) 
           sceneCol *= 1. - edge2*.6; // Bump mapped edging for the terrain only.     
        
        //if(svObjID>.5) 
        sceneCol *= 1. - edge*.8; // Geometry based edging.
        
        // Reflection. Not really suitable for this example.
        //sceneCol += eMap(reflect(rd, sn), sn);

        // Apply the attenuation and shadows.
        sceneCol *= atten*sh;
    
    }
    
  
    // Return the color. Done once for each pass.
    return sceneCol;
    
}




         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin


    // Screen coordinates.
//    float2 uv = (fragCoord - iResolution.xy*.5)/iResolution.y;
    
    // Camera Setup.
//    float3 ro = float3(0, 0, _Time.y*5.); // Camera position, doubling as the ray origin.
    float3 lk = ro + float3(0, 0, .25);  // "Look At" position.
 
   
    // Light position. Set reasonably far away in the background somewhere. A sun is usually so far 
    // away that direct light is called for, put I like to give it just a bit of a point light feel.
    float3 lp = ro + float3(10., FAR*.24, FAR*.52)*3.;
    
   
    // Using the Z-value to perturb the XY-plane.
    // Sending the camera, "look at," and light floattor down the path. The "path" function is 
    // synchronized with the distance function.
//    ro.xy += path(ro.z);
//    lk.xy += path(lk.z);
//    lp.xy += path(lp.z);
    

    // Using the above to produce the unit ray-direction floattor.
    float FOV = 3.14159/3.; // FOV - Field of view.
    float3 forward = normalize(lk-ro);
    float3 right = normalize(float3(forward.z, 0., -forward.x )); 
    float3 up = cross(forward, right);

    // rd - Ray direction.
//    float3 rd = (forward + FOV*uv.x*right + FOV*uv.y*up);
//    rd = normalize(float3(rd.xy, sqrt(max(rd.z*rd.z - dot(rd.xy, rd.xy)*.15, 0.)) ));
    
    // Camera swivel - based on path position.
    float2 sw = path(lk.z);
//    rd.xy = mul(rd.xy,r2(-sw.x/32.));
//    rd.yz = mul(rd.yz,r2(-sw.y/16.));

    //rd coord system XYZ X=+right/-left  Z=+fwd/-back Y=+up/-down
    rd = (screenUV.x<.002 &&  screenUV.y<.01)?float3(0,0,-1) :rd;
    
    // Retrieve the background color.
    float3 sky = getSky(ro, rd, lp);
    
    
    // Trace the scene.    
    float t = trace(ro, rd);
    
    svObjID = objID; // Save the object ID, for use in the coloring equation.
    
    
    // Retrieving the color at the initial hit point.
    float3 sceneColor = doColor(ro, rd, lp, t);
         
    
    // APPLYING FOG
    // Fog - based off of distance from the camera.
    float fog = smoothstep(0., 1, t/FAR*.25); // t/FAR; 

    // Blend in a bit of light fog for atmospheric effect. I really wanted to put a colorful, 
    // gradient blend here, but my mind wasn't buying it, so dull, blueish grey it is. :)
    float3 fogCol = sky;//lerp(float3(.6, .9, 1).zyx, float3(.62, .68, 1).zyx, rd.y*.5 + .5)*1.25;
    sceneColor = lerp(sceneColor, fogCol, fog); // exp(-.002*t*t), etc. fog.zxy 
    
    
    // POSTPROCESSING
    
    // Sprinkles.
    //sceneColor *= 1. + hash31(sp)*.1 - .05; 

    
    // Subtle vignette.
//    uv = fragCoord/iResolution.xy;
//    sceneColor *= pow(16.*uv.x*uv.y*(1. - uv.x)*(1. - uv.y) , .125)*.75 + .25;
    // Colored varation.
    //sceneColor = lerp(pow(min(float3(1.5, 1, 1)*sceneColor, 1.), float3(1, 3, 16)), sceneColor, 
                     //pow( 16.0*uv.x*uv.y*(1.0-uv.x)*(1.0-uv.y) , .125)*.5 + .5);
    /*
    // A very simple overlay. Two linear waves - rotated at 60 degree angles - to give a dot-matrix vibe.
    uv = sin(uv*r2(3.14159/6.)*3.14159*iResolution.y/1.5)*.1 + 1.;
    sceneColor *= uv.x*uv.y;
    */
    // Mild LCD overlay.
    //float2 rg = lerp(mod(fragCoord, float2(3))*sceneColor.xy, sceneColor.xy, .65);
    //sceneColor = float3(rg, sceneColor.z - lerp(sceneColor.x - rg.x, sceneColor.y - rg.y, .65));
    
   

    // Clamping the scene color, then presenting to the screen.
    fragColor = (screenUV.x<.002 &&  screenUV.y<.01)?float4(t*.1,0,0,1):float4(sqrt(clamp(sceneColor, 0.0, 1.0)), 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}





//  https://www.shadertoy.com/view/ldjyzc

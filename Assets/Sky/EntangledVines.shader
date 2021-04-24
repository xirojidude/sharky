
Shader "Skybox/EntangledVines"
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
            #define mat2 float2x2
            #define textureLod(a,b,c) tex2Dlod(a,float4(b,0,c))
            #define atan(a,b) atan2(b,a)

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


/*
    Entangled Vines
    ---------------
    
    Entangled vine fly-through, which is nothing more than a mutated lattice traversal in disguise.
    
    I wrote this on a pretty slow computer, so had to do everything on the cheap, which gave me a
    good excuse to use Nimitz's Log-Bisection method. I swear I'm not stalking his code, but it seems 
    every time I need a solution to something, one of his examples magically provides the answer. :)

    The rest was written off the top of my head, but is an amalgamation of a whole bunch of things 
    I've encountered on Shadertoy at one time or another. The simple scene itself was constructed 
    with Gyabo's "Raymarching Basic" in mind. Although I've used different methods, I'm pretty sure 
    Dave Hoskins's "Skin Peeler" (based on Xyptonjtroz) influenced the coloring style. The misty 
    overlay is loosely infuenced from the same example.
    
    Related examples:
    
    Log-Bisection Tracing - Nimitz
    https://www.shadertoy.com/view/4sSXzD
    
    Skin Peeler - Dave Hoskins
    https://www.shadertoy.com/view/XtfSWX
    Based on Xyptonjtroz: Nimitz
    https://www.shadertoy.com/view/4ts3z2
    
    Raymarching Basic - Gyabo
    https://www.shadertoy.com/view/Ml2XRD    

*/

#define FAR 30.
#define PI 3.14159265

// Grey scale.
float getGrey(vec3 p){ return p.x*0.299 + p.y*0.587 + p.z*0.114; }

// 2x2 rotation matrix.
mat2 rot(float th) {
    float cs = cos(th), sn = sin(th); return mat2(cs, -sn, sn, cs);
}

vec3 firePalette(float i){

    float T = 1400. + 1300.*i; // Temperature range (in Kelvin).
    vec3 L = vec3(7.4, 5.6, 4.4); // Red, green, blue wavelengths (in hundreds of nanometers).
    L = pow(L,vec3(5.0,5,5)) * (exp(1.43876719683e5/(T*L))-1.0);
    return 1.0-exp(-5e8/L); // Exposure level. Set to "50." For "70," change the "5" to a "7," etc.
}


// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch01.html
vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){
   
    //n = abs(n)/1.732051;
    n = max((abs(n) - 0.2)*7., 0.001); // n = max(abs(n), 0.001), etc.
    //n = max(abs(n), 0.001);
    n /= (n.x + n.y + n.z );  
    
    return (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
}


// Hash to return a scalar value from a 3D vector.
float hash31(vec3 p){ return fract(sin(dot(p, vec3(127.1,311.7, 74.7)))*43758.5453); }

// Standard hash algorithm that you'll see all over the place.
vec3 hash33(vec3 p) { 

    // Faster, but doesn't disperse things quite as nicely as the block below it. However, when framerate
    // is an issue, and it often is, this is the one to use. Basically, it's a tweaked amalgamation I put
    // together, based on a couple of other random algorithms I've seen around... so use it with caution,
    // because I make a tonne of mistakes. :)
    float n = sin(dot(p, vec3(7, 157, 113)));
    return fract(vec3(2097152, 262144, 32768)*n)*2.-1.;  
    //p = fract(vec3(2097152, 262144, 32768)*n); 
    //return sin(p*TAU + time); // Cheap...ish animation.

}

// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total.
vec3 doBumpMap( sampler2D tex, in vec3 p, in vec3 nor, float bumpfactor){
   
    const float eps = 0.001;
    vec3 grad = vec3( getGrey(tex3D(tex, vec3(p.x-eps, p.y, p.z), nor)),
                      getGrey(tex3D(tex, vec3(p.x, p.y-eps, p.z), nor)),
                      getGrey(tex3D(tex, vec3(p.x, p.y, p.z-eps), nor)));
    
    grad = (grad - getGrey(tex3D(tex,  p , nor)))/eps; 
            
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
    
}

// Smooth minimum function. Hardcoded with the smoothing value "0.25."
float sminP(in float a, in float b ){
    
    float h = clamp(2.*(b - a) + 0.5, 0.0, 1.0);
    return (b - 0.25*h)*(1. - h) + a*h;
    
}

// The vine structure.
// 
// I had Shadertoy user Gyabo's "Raymarching Basic" in mind when putting this together:
// https://www.shadertoy.com/view/Ml2XRD
// His example Cune is also worth looking at: https://www.shadertoy.com/view/ls2GDR
float map(vec3 p) {
    
    // Basic idea: Create a bunch of repeated warped cylinders aligned along the X, Y 
    // and Z directions whilst offsetting their positions. Find their minimum, then 
    // add some bumps.
    //
    // 
    // "length(p.xy) - r" is a standard way to create a cylinder in the Z-direction.
    // Performing "mod(p.xy, a) - a/2." will create multiple cylinders.
    // Perturb the XY-plane to make it windy, and add an offset, while your at it.
    // Do the same in the X and Y directions.
    vec2 perturb = vec2(sin((p.z * 2.15 + p.x * 2.35)), cos((p.z * 1.15 + p.x * 1.25)));
    vec2 perturb2 = vec2(cos((p.z * 1.65 + p.y * 1.75)), sin((p.z * 1.4 + p.y * 1.6)));
    vec2 q1 = mod(p.xy + vec2(0.25, -0.5), 2.) - 1.0 + perturb*vec2(0.25, 0.5);
    vec2 q2 = mod(p.yz + vec2(0.25, 0.25), 2.) - 1.0 - perturb*vec2(0.25, 0.3);
    vec2 q3 = mod(p.xz + vec2(-0.25, -0.5), 2.) - 1.0 - perturb2*vec2(0.25, 0.4);
    
    // Used to add some bumps on the overall structure. See "p.x*p.y*p.z.. etc" below.
    p = sin(p*8. + cos(p.yzx*8.));
    
    // The cylinders.
    float s1 = length( q1 ) - 0.24; // max(abs(q1.x), abs(q1.y)) - 0.2; // etc.
    float s2 = length( q2 ) - 0.24;
    float s3 = length( q3 ) - 0.24;
    
    /*
    // These are cheaper. Unfortunately, the raymarching function takes longer to find the
    // hit point, so that can be problematic.
    float s1 = dot( q1, q1 ) - 0.06; 
    float s2 = dot( q2, q2 ) - 0.06;
    float s3 = dot( q3, q3 ) - 0.06;
    */
    
    // Retrieve the minimum of s1, s2 and s3, then add the bumps.
    // In this case, a smooth minimum is used, because it looks a little more organic.
    return sminP(sminP(s1, s3), s2) - p.x*p.y*p.z*0.05;
    
}

// The iterations should be higher for proper accuracy, but in this case, the shadows are a subtle background feature, so
// hopefully, it's not too noticeable.
float softShadow(in vec3 ro, in vec3 rd, in float start, in float end, in float k){

    float shade = 1.0;
    const int maxIterationsShad = 12; // 24 or 32 would be better. Even 16 would be good, but my computer says, "No." :)

    // The "start" value, or minimum, should be set to something more than the stop-threshold, so as to avoid a collision with 
    // the surface the ray is setting out from. It doesn't matter how many times I write shadow code, I always seem to forget this.
    // If adding shadows seems to make everything look dark, that tends to be the problem.
    float dist = start;
    float stepDist = end/float(maxIterationsShad);

    // Max shadow iterations - More iterations make nicer shadows, but slow things down. Obviously, the lowest 
    // number to give a decent shadow is the best one to choose. 
    for (int i=0; i<maxIterationsShad; i++){
        // End, or maximum, should be set to the distance from the light to surface point. If you go beyond that
        // you may hit a surface not between the surface and the light.
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist));
        
        // What h combination you add to the distance depends on speed, accuracy, etc. To be honest, I find it impossible to find 
        // the perfect balance. Faster GPUs give you more options, because more shadow iterations always produce better results.
        // Anyway, here's some posibilities. Which one you use, depends on the situation:
        // +=max(h, 0.001), +=clamp( h, 0.01, 0.25 ), +=min( h, 0.1 ), +=stepDist, +=min(h, stepDist*2.), etc.
        
        // In this particular instance the light source is a long way away. However, we're only taking a few small steps
        // toward the light and checking whether anything "locally" gets in the way. If a part of the vine a long distance away
        // is between our hit point and the light source, it won't be accounted for. Technically that's not correct, but the local
        // shadows give that illusion... kind of.
        dist += clamp(h, 0.0001, .2);
        
        // There's some accuracy loss involved, but early exits from accumulative distance function can help.
        if (abs(h)<0.001 || dist > end) break; 
    }

    // I usually add a bit to the final shade value, which lightens the shadow a bit. It's a preference thing. Really dark shadows 
    // look too brutal to me.
    return min(max(shade, 0.) + 0.4, 1.0); 
}

// Standard ambient occlusion. Based on the original by IQ.
float calculateAO(vec3 p, vec3 n){

    const float AO_SAMPLES = 5.0;
    float r = 0.0, w = 1.0, d;
    
    for (float i=1.0; i<AO_SAMPLES+1.1; i++){
        d = i/AO_SAMPLES;
        r += w*(d - map(p + n*d));
        w *= 0.5;
    }
    
    return 1.0-clamp(r,0.0,1.0);
}

// Tetrahedral normal: I remember a similar version on "Pouet.net" years ago, but this one is courtesy of IQ.
// I'm using it, because I'm trying to make as few surface function calls (4 versus 6) as possible.
vec3 getNormal( in vec3 p ){

    vec2 e = vec2(0.5773,-0.5773)*0.001;
    return normalize( e.xyy*map(p+e.xyy ) + e.yyx*map(p+e.yyx ) + e.yxy*map(p+e.yxy ) + e.xxx*map(p+e.xxx ));
}



// Log-Bisection Tracing
// https://www.shadertoy.com/view/4sSXzD

// Log-Bisection Tracing by nimitz (twitter: @stormoid)
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Contact: nmz@Stormoid.com

// Notes: I've made some minor changes to Nitmitz's original to suit this particular example. It 
// seems to work, but if you're interested in the function itself, I'd probably bypass this example, 
// and refer to the original function in the link above. There, you'll find a good explanation as to 
// how it works too.

// In essence, the first loop manages to hone in on the hit point pretty quickly, mainly due to the
// line involving the logarithm (clever). However, it has a tendency to overshoot. If that happens, 
// points from either side of the hit point are fed into the second loop, which employs a bisection 
// method that eventually gets you there. There's a bunch of stuff that I'm skipping over, and you 
// can read about it in the original. The function doesn't work for all setups, but in cases such as 
// this, you can get a pretty decent performance boost.

#define PRECISION 0.001

float logBisectTrace( in vec3 ro, in vec3 rd){


    float t = 0., told = 0., mid, dn;
    float d = map(rd*t + ro);
    float sgn = sign(d);

    for (int i=0; i<80; i++){

        // If the threshold is crossed with no detection, use the bisection method.
        // Also, break for the usual reasons. Note that there's only one "break"
        // statement in the loop. I heard GPUs like that... but who knows?
        if (sign(d) != sgn || d < PRECISION || t > FAR) break;
 
        told = t;
        
        // If the distance from the root is high enough, use standard raymarching,
        // instead of the log bisect method.
        //if (d > 1.) t += d*.75;
        // "0.75" determines how fast the root finder moves in. Needs to be lowered when dealing 
        // with thin slices. The potential problem is the intersector crossing the function twice 
        // in one step.
        //else t += log(abs(d) + 1.1)*.75; // //To cross faster, a minimum step size.
        
        // Branchless version of the above.        
        t += step(-1., -d)*(log(abs(d) + 1.1)*.7 - d*.7) + d*.7;
        
        d = map(rd*t + ro);
    }
    
    // If a threshold was crossed without a solution, use the bisection method.
    if (sign(d) != sgn){
    
        // Based on suggestions from CeeJayDK, with some minor changes.

        dn = sign(map(rd*told + ro));
        
        vec2 iv = vec2(told, t); // Near, Far

        // 6 iterations seems to be more than enough, for most cases...
        // but there's an early exit, so I've added a couple more.
        for (int ii=0; ii<8; ii++){ 
            //Evaluate midpoint
            mid = dot(iv, vec2(.5,.5));
            float d = map(rd*mid + ro);
            if (abs(d) < PRECISION)break;
            // Suggestion from movAX13h - Shadertoy is one of those rare
            // sites with helpful commenters. :)
            // Set mid to near or far, depending on which side we're on.
            iv = mix(vec2(iv.x, mid), vec2(mid, iv.y), step(0.0, d*dn));
        }

        t = mid; 
        
    }
    
    //if (abs(d) < PRECISION) t += d;

    return t;
}



/////
// Code block to produce four layers of fine mist. Not sophisticated at all.
// If you'd like to see a much more sophisticated version, refer to Nitmitz's
// Xyptonjtroz example. Incidently, I wrote this off the top of my head, but
// I did have that example in mind when writing this.
float trig3(in vec3 p){
    p = cos(p*2. + (cos(p.yzx) + 1. + iTime*4.)*1.57);
    return dot(p, vec3(0.1666,0.1666,0.1666)) + 0.5;
}

// Basic low quality noise consisting of three layers of rotated, mutated 
// trigonometric functions. Needs work, but it's OK for this example.
float trigNoise3D(in vec3 p){

    // 3D transformation matrix.
    const mat3 m3RotTheta = mat3(0.25, -0.866, 0.433, 0.9665, 0.25, -0.2455127, -0.058, 0.433, 0.899519 )*1.5;
  
    float res = 0.;

    float t = trig3(p*PI);
    p += (t - iTime*0.25);
    p = mul(m3RotTheta,p);
    //p = (p+0.7071)*1.5;
    res += t;
    
    t = trig3(p*PI); 
    p += (t - iTime*0.25)*0.7071;
    p = mul(m3RotTheta,p);
     //p = (p+0.7071)*1.5;
    res += t*0.7071;

    t = trig3(p*PI);
    res += t*0.5;
     
    return res/2.2071;
}

// Four layers of cheap trigonometric noise to produce some subtle mist.
// Start at the ray origin, then take four samples of noise between it
// and the surface point. Apply some very simplistic lighting along the 
// way. It's not particularly well thought out, but it doesn't have to be.
float getMist(in vec3 ro, in vec3 rd, in vec3 lp, in float t){

    float mist = 0.;
    ro += rd*t/8.; // Edge the ray a little forward to begin.
    
    for (int i = 0; i<4; i++){
        // Lighting. Technically, a lot of these points would be
        // shadowed, but we're ignoring that.
        float sDi = length(lp-ro)/FAR; 
        float sAtt = min(1./(1. + sDi*0.25 + sDi*sDi*0.05), 1.);
        // Noise layer.
        mist += trigNoise3D(ro*2.)*sAtt;
        // Advance the starting point towards the hit point.
        ro += rd*t/4.;
    }
    
    // Add a little noise, then clamp, and we're done.
    return clamp(mist/2. + hash31(ro)*0.1-0.05, 0., 1.);

}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.01+ _XYZPos;                                             // ray origin

    // Screen coordinates.
    vec2 uv = (fragCoord.xy - iResolution.xy*0.5)/iResolution.y;

    // Ray origin. Traversing with time along the Z-axis.
//    vec3 ro = vec3(0., 0., iTime*2.);
    
    // Lazy way to construct a unit direction ray.
//    vec3 rd = normalize(vec3(uv, 0.5));
    
    // Equally lazy way to look around the scene by rotating the unit direction vector. 
//    mat2 m2 = rot(iTime * 0.25);
//    rd.xz *= m2;
//    rd.xy *= m2;
//    rd.yz *= m2;
    

    // The light position. In this case, it's the quasi-distant sun position, which is situated about 30 units in front
    // of the viewing position, "ro." I've arranged for it to rotate about its postion just a little.
    // "rd*10." is not realistic. No distant light source rotates around like that, but it makes the shadows move a little.
    // Obviously, a real sun would be much, much further away, but by keeping it within a workable distance, you can
    // get a bit of a point light effect.
    vec3 lp = vec3(0., 0., FAR) + ro + rd*10.;
    
    // Standard way to put a light in the sky. Dot the unit direction vector with the unit light direction vector.
    // Normalize the result, then ramp up the power. In this case, I want to spread the brightness out, so a lower figure 
    // of about "4" is being chosen. For a more contrasty sky with intense sun, larger values are used. 
    float bgShade = pow(max(dot(rd, normalize(lp - ro)), 0.)*0.5+0.5, 4.);
    // Background (or sky) color. Blend the shade between two colors. These two aren't very inspiring, but the coloring
    // is being done post process, so it's essentially dark to bright for now.
    vec3 bc = mix(vec3(.0, .0,.05), vec3(1.,1,1), bgShade);
    
    // Initiate the scene color to the background (sky) color.
    vec3 sc = bc;
    
    // Use Nimitz's really fancy raymarching algorithm. :)
    float t = logBisectTrace(ro, rd);

    
    if(t<FAR){
        
        // Surface position.
        vec3 sp = ro + rd*t;
        
        // Normal.
        vec3 sn = getNormal(sp);
        
        // Texture bump the normal.
        const float texSize0 = 1./2.;
        sn = doBumpMap(_MainTex, sp*texSize0, sn, 0.025);
        
        // Obtain the texel color at the surface position.
        vec3 objCol = tex3D( _MainTex, sp*texSize0, sn );//vec3(min(c*1.5, 1.), pow(c, 2.5), pow(c, 10.));
        
        // Light direction vector. From the sun to the surface point.
        vec3 ld = lp-sp;
        
        // Distance from the surface postion to the light source (sun position).
        float lDist = max(length(ld), 0.001);
        
        ld /= lDist; // Normalize the light direct vector.
        
        // Attenuation, based on the distance of the light (sun) to the surface point.
        lDist /= FAR; // Bringing the light distance down to the zero to one range, which is more workable.
        float sAtten = min(1./(1. + lDist*0.125 + lDist*lDist*0.05), 1.);
        
        
        // Shadowing and occlusion. 
        float shad = softShadow(sp, ld, 0.05, FAR, 8.);
        float ao = calculateAO(sp, sn);
        
        // Standard diffuse and specular calculations.
        float diff = max(dot(sn, ld), 0.);
        float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 8.);
       
        // Combining the properties above to produce the lit color.
        sc = (objCol*(diff + 0.5) + spec)*sAtten;
       
        // Applying the shadows and occlusion.
        sc = min(sc, 1.)*shad*ao;
        
    
    }
    
    
    // Fog.
    //
    // Fog - Based on distance from the viewing position. Not to be confused with the misty haze.
    // Mix the background color (sky color above) and the object color according to a falloff value,
    // which is analogous to fog, so we call it that. Pretty standard.
    float fog = min(1.0 / (1. + t*0.25 + t*t*0.025), 1.);
    sc = mix(bc, sc, fog);
    //sc = mix(sc, bc, smoothstep(0.0, FAR-20., t)); // Another way to mix things, but using a quick transition.
    
    
    // Color post processing. Fading from orange to the original color... for no particular reason.
    // Comment the following block out to see the original effect minus the orange.
    vec3  sc2 = firePalette(getGrey(sc));
    float fadeFactor = min(1.0 / (1. + t), 1.); // Color fade factor. Made up to suit the conditions.
    sc = mix(sc, sc2, fadeFactor*0.34+0.66);
    
    
    // Fake misty overlay.  
    //
    // Adding the misty haze... otherwise known as the lamest volumetric effect ever. :)
    // Start at the ray origin, then accumulate four layers between it and the hit position.
    float mist = getMist(ro, rd, lp, t);
    
    // Combining the mist value, sky color (bgShade, etc) and fog to give a fog color.
    // Part science, part made up.
    vec3 fogCol = mix(vec3(bgShade*0.8+0.2,bgShade*0.8+0.2,bgShade*0.8+0.2)*vec3(1., 0.85, 0.6), sc, mist*fog);
    // Toning down the fog color. Totally made up. :)
    sc = sc*0.65 + fogCol*0.35;
     
    
    // Done.
    fragColor = vec4(clamp(sc, 0., 1.), 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}

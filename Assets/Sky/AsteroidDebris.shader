
Shader "Skybox/AsteroidDebris"
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

/*

    Asteroid Debris
    ---------------

    Packing a few 3D distance fields into one face of the cubemap to raymarch
    an asteroid field in realtime.
    
    In regard to the scene itself, I'm basically paying hommage to IQ's Leizex 
    demonstration, which has always been a favorite of mine -- The link is below,
    for anyone who hasn't seen it. The main difference between this example and
    IQ's is that I have the benefit of coding on a system from over a decade 
    after he made it. The inspiration to do this in the first place came from
    TDHooper's awesome "Copper \ Flesh" example -- The link to that is below also.

    As far as the physics engine and lighting goes... Well, it doesn't. It's all 
    made up, so I'd pretty much ignore that part of it. :D The main focus of this
    demonstration was to fill up a 100 sided voxel cube with distance equations, 
    pack them into one of the cubemap faces, them read them back and render them.

    It's almost 2020, and computers have crazy power compared to those 20 years
    ago. However, raymarching surfaces like this at reasonable frame rates would 
    still be impossible without precalculation. Curiosity got the better of me,
    so I dropped a less expensive version of this particular surface that was 
    calculated on the fly into the raymarching loop and even that virtually 
    brought my Alienware machine to a grinding halt. In fact, I have a feeling 
    that trying to calculate these fields on the fly as part of a static image 
    inside a pathtracing demonstration would bring about compiler issues.

    There are 3 surfaces here: The first is a couple of layers of 3D gradient 
    noise -- A few years ago, that surface alone would have been a bit much.
    The second and third layers consist of insanely expensive 3D plane-based 
    rounded Voronoi implementations, which involved exponentials and well over 
    a hundred passes each. 

    There are still a few technicalities you have to overcome. The storage size
    on the cube map is fantastic - There are six 1024 by 1024 RGBA face textures.
    However, you still need to make a choice between resolution and speed. Within
    reason, you can take as long as you want to construct your surfaces and pack
    them into the cube map. However, reading them out in realtime needs to be 
    efficient. For now, I've taken the easy route and packed in a cube with 100
    pixels for each of the XYZ positions -- It's an obvious choice, since 100
    cubed is a million pixels which is just shy of 1024 x 1024. This means one
    surface per channel, or three surfaces per texel read -- The last channel is 
    used to store resolution for screen size changes, etc, since the "iFrame"
    variable is not reliable -- Well, not for me anyway.

    Looking up a single voxel value when reconstructing the surface won't cut it,
    so an interpolation of 8 neighboring voxels is necessary, which means 8 texel
    reads inside the distance function. My machine can do that with ease, but in
    the future, I'll have to cut it down, and I know of of a few ways to do that.

    Anyway, this was just a simple demonstration to get something on the board,
    as they say. I have more interesting examples coming. As for improvements to
    be made, there are too many to name. One obvious one, is making better use
    of the cubemap, since I'm using just a small portion of it. You could use all 
    four channels in one 1024 by 1024 face texture, and have enough room for a 
    cube with side dimensions of 160 pixels. Using all four channels would add to 
    the technicality, but I think I know of a way to bring the rendering side 
    down from 8 interpolated texture reads to just 3, or even two, but don't 
    quote me on that. :)


   
    Inspired by:

    // Really nice example, and the thing that motivated me to get in amongst it
    // and finally learn to read and write from the cube map. I have a few 3D 
    // examples coming up, which use more simplistic formulae, but I couldn't tell 
    // you whether that translates to extra speed or not. Knowing how I code, 
    // probably not. :D
    Copper / Flesh - tdhooper
    https://www.shadertoy.com/view/WljSWz


    Loosely based on:
    
    // I used to marvel at this a few years ago. Even today, it's a
    // nice looking piece of imagery.
    Leizex (made in 2008) - iq
    https://www.shadertoy.com/view/XtycD1


    2D usage of a cubemap:

    // Using a 3D feature to do something in 2D seems counter intuitive. However,
    // I find that one cubemap face is more useful for 2D storage than the in-house
    // buffers. By the way, Fabrice was already using it for this purpose before
    // it was cool. :D
    Turing Texture - Shane
    https://www.shadertoy.com/view/WsGSDR


*/


// Far distance, or not very far, in this case. :)
#define FAR 6.


// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch01.html
vec3 tex3D(sampler2D t, in vec3 p, in vec3 n){
    
    // We only want positive normal weightings. The normal is manipulated to suit
    // your needs.
    n = max(n*n - .2, .001); // n = max(abs(n) - .1, .001), etc.
    //n /= dot(n, vec3(1)); // Rough renormalization approximation.
    n /= length(n); // Renormalizing.
    
    vec3 tx = texture(t, p.yz).xyz; // Left and right sides.
    vec3 ty = texture(t, p.zx).xyz; // Top and bottom.
    vec3 tz = texture(t, p.xy).xyz; // Front and back.
    
    // Blending the surrounding textures with the normal weightings. If the surface is facing
    // more up or down, then a larger "n.y" weighting would make sense, etc.
    //
    // Textures are stored in sRGB (I think), so you have to convert them to linear space 
    // (squaring is a rough approximation) prior to working with them... or something like that. :)
    // Once the final color value is gamma corrected, you should see correct looking colors.
    return (tx*tx*n.x + ty*ty*n.y + tz*tz*n.z);
    
}

// Camera path.
vec2 path(float t) { return sin(t*vec2(1, .8)) + sin(t*vec2(.3, .2)); }


// Basically the same as what I normally do, but I got this snippet from 
// unnick's "light at the end of the tunnel" example, here:
// https://www.shadertoy.com/view/WsySR1
//
// Camera view matrix.
mat3 getView(float t){
    
    const float FOV = 2.;
    vec3 fwd = normalize(vec3((path(t + .1) - path(t))/.1, FOV));
    vec3 up = vec3(0., 1, 0) - fwd*fwd.y;
    vec3 rgt = cross(fwd, up);
    return transpose(mat3(rgt, -up, fwd));
}

// Hacky global to record the glow variable to be using in the "trace" function.
vec3 glow3;


// The scene function.
float map(vec3 p) {
    
    
    // Gaz's path correction. Very handy.
    vec2 pth = path(p.z);
    vec2 dp = (path(p.z + .1) - pth)/.1; // Incremental path diffence.
    vec2 a = cos(atan(dp)); 
    // Wrapping a tunnel around the path.
    float tun = length((p.xy - pth)*a);
    
    
    
    // Obtaining the distanc field values from the 3D data packed into
    // the cube map face. These have been smoothly interpolated.
    vec3 tx3D = texMapSmooth(iChannel0, p/3.).xyz;
    // Using this will show you why interpolation is necessary.
    //vec3 tx3D = tMap(iChannel0, p/3.).xyz;
    
    // The main surface. Just a couple of gradient noise layers. This is used
    // as a magnetic base to wrap the asteroids around.
    float main = (tx3D.x - .55)/2.;
    
    // Calling the function again, but at a higher resolution, for the other
    // surfaces, which consist of very expensive rounded Voronoi.
    tx3D = texMapSmooth(iChannel0, p*2.).xyz;
    
    // Saving the higher resolution gradient noise to add some glow. I patched
    // this in at the last minute.
    glow3 = tx3D;

    
    // Attaching the asteroid field to the gradient surface. Basically, the 
    // rocks group together in the denser regions. With doing this, you'd 
    // end up with a constant density mass of rocks.
    main = smax(main, -(tx3D.z + .05)/6., .17);
    
    // Adding a heavy layer of gradient noise bumps to each rock.
    main += (abs(tx3D.x - .5)*2. - .15)*.04;
   
    // Smoothly running the tunnel through the center, to give the camera
    // something to move through -- Otherwise, it'd bump into rocks. Getting 
    // a tunnel to run through a group of rocks without warping them was 
    // only possilbe because of the way the rocks have been constructed.
    return smax(main, -tun, .25);
    
}


// Surface bump function.
float bumpSurf3D( in vec3 p){
    
    
    // Obtaining the distanc field values from the 3D data packed into
    // the cube map face. These have been smoothly interpolated.
    vec3 tx3D = texMapSmooth(iChannel0, p*6.).xyz;
    
    // This is a mixed rounded Voronoi surface, moulded to look a bit lit
    // holes or craters for that pitted look.
    return tx3D.y;
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

/*
// Cheap shadows are hard. In fact, I'd almost say, shadowing particular scenes with limited 
// iterations is impossible... However, I'd be very grateful if someone could prove me wrong. :)
float softShadow(vec3 ro, vec3 lp, vec3 n, float k){

    // More would be nicer. More is always nicer, but not really affordable... Not on my slow test machine, anyway.
    const int maxIterationsShad = 32; 
    
    ro += n*.002;
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
        t += clamp(d, .025, .35); 
        
        
        // Early exits from accumulative distance function calls tend to be a good thing.
        if (d<0. || t>end) break; 
    }

    // Sometimes, I'll add a constant to the final shade value, which lightens the shadow a bit --
    // It's a preference thing. Really dark shadows look too brutal to me. Sometimes, I'll add 
    // AO also just for kicks. :)
    return max(shade, 0.); 
}
*/
        
// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float calcAO(in vec3 p, in vec3 n)
{
    float sca = 4., occ = 0.;
    for( int i = 0; i<5; i++ ){
    
        float hr = float(i + 1)*.15/5.;        
        float d = map(p + n*hr);
        occ += (hr - d)*sca;
        sca *= .7;
    }
    
    return clamp(1. - occ, 0., 1.);  
    
}


 
// Standard normal function. It's not as fast as the tetrahedral calculation, but more symmetrical.
vec3 normal(in vec3 p, float ef) {
    vec2 e = vec2(.001*ef, 0);
    return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
}

// Hacky global for the glow.
float glow;

// Basic raymarcher.
float trace(in vec3 ro, in vec3 rd, in vec3 lp){

    float t = 0., d;
    
    float l = length(lp - ro);
    
    glow = 0.;
    
    for(int i = min(iFrame, 0); i<128; i++){
    
        d = map(ro + rd*t);
        // Note the "t*b + a" addition. Basically, we're putting less emphasis on accuracy, as
        // "t" increases. It's a cheap trick that works in most situations... Not all, though.
        if(abs(d)<.001*(t*.1 + 1.) || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.), etc.
        
        
        //float l2D = l;//length(l2 - pp);
        // Distance based falloff.
        //float aD = abs(d);
        //if(aD<.15) glow += 1.*(.15 - aD)/(.0001 + l2D*l2D);
        // Accumulating the attenuation, whilst applying a little noise.
        //
        glow += .04/(.0001 + l)*glow3.x;//
        
        // In an ideal world, we'd only render functions with Lipschitz contants within
        // acceptable ranges (less than one, I think), but with anything interesting,
        // it rarely happens. Therefore, hacks like ray shortening are necessary.
        t += d*.7;
    }
    

    // Clip the distance to the maximum to avoid artifacts. 
    return min(t, FAR);
}

 
void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    
    
    // Aspect correct screen coordinates.
    vec2 uv = (fragCoord - iResolution.xy*.5)/iResolution.y;
    
    
    // Slight barrel distortion.
    float r = dot(uv, uv);
    uv *= 1. + .15*(r*r + r);
    
    // Time for the camera and light, viewing matrix and
    // the unit direction ray.
    vec2 tm = vec2(iTime/2.) + vec2(0, 1.25);
    mat3 vMat = getView(tm.x);
    vec3 rd = normalize(vec3(-uv, 1.))*vMat;
    
    // Moving the camera and the light along the path.
    vec3 ro = vec3(path(tm.x), tm.x);
    vec3 lp = vec3(path(tm.y), tm.y);
    
    
    // Raymarch the scene.
    float t = trace(ro, rd, lp);
    
    // Initiate the scene color to zero.
    vec3 col = vec3(0);
    
    // If we've hit something, light it up.
    if (t<FAR){
        
        
        // Surface postion and the normal at that position.
        vec3 sp = ro + rd*t;
        vec3 sn = normal(sp, 1.);
        
        
         
        // Function based bump mapping.
        sn = doBumpMap(sp, sn, .01);///(1. + t*t/FAR/FAR*.25)
      
        
       
        // Trusty "Rusty Metal" texture -- I'm trying to set a Shadertoy
        // record for its overusage. :D
        vec3 tx = tex3D(iChannel1, sp*2., sn);
        tx = smoothstep(-.1, .5, tx);
        
        // Set the asteroid color to the texture color.
        vec3 oCol = tx*2.;
        
        
        // Applying some extra shading to the bumped craters.
        float ns = bumpSurf3D(sp);
        oCol *= ns*1.5;
       
       
    
        // I noticed that the soft shadows weren't contributing enough to the scene, so 
        // figured, I may as well go with something cheaper. Just for the record, this
        // example still runs pretty quickly with them as well.
        
        // Needs to match the frequency of that in the distance function.
        vec3 tx3D = texMapSmooth(iChannel0, (sp/3.)).xyz;
        //
        //float sh = softShadow(sp, lp, sn, 8.);
        float sh = tx3D.x;
        float ao = calcAO(sp, sn); // Ambient occlusion.
        // Ramping up the ambient occlusion, since they're now faking the job of the light
        // shadows as well.
        ao *= ao; 
        sh = min(sh + ao*.3, 1.);


        // Surface to light vector, corresponding distance, then normalizing "ld."
        vec3 ld = lp - sp;
        float lDist = max(length(ld), .001);
        ld /= lDist;

        // Light and distance attenuation: The scene can get a little speckly in the distance, 
        // so I've attenuated the light that reaches the viewer with respect to distance also.
        // I don't often do it, but it seemed necessary here.
        float atten = 1./(1. + lDist*lDist*.25)/(1. + t);
       

        // Diffuse, specular and Fresnel calculations.
        float dif = max(dot(ld, sn), 0.); // Diffuse term.
        float spe = pow(max( dot( reflect(-ld, sn), -rd ), 0.), 32.); // Specular term.
        float fre = clamp(1.0 + dot(rd, sn), 0., 1.); // Fresnel reflection term.
        
        
        /*
        // Electric charge: I could not make this work, but at least I tried, so I get
        // participation award, right? :D I'll tweak it later, and see whether I can make 
        // it work. :)
        float hi = smoothstep(.0, .1, abs(n3D(sp + vec3(0, 0, iTime)) - .5)*2.);
        oCol += fre*glow*vec3(.1, 1, .7)*.25/(.01 + hi*hi)/(1. + t*.25);
        */
 
        // Applying the above to produce a color.
        col = oCol*(dif*vec3(4, .8, .5) + vec3(2, .7, .4)*spe + vec3(.1, .3, 1)*fre + ao*.2);
        
        
        // A bit of reflection. I put this in as an afterthought.
        col += col*vec3(1, .7, .3)*fBm2(32.*reflect(rd, sn));
 
       
        // Applying the ambient occlusion, very fake shadow term and light attenuation.
        col *= ao*sh*atten;
   
    }
    
    // Combine the scene with a gradient fog color.
    vec3 sky = mix(vec3(1), vec3(.6, .8, 1), -rd.y*.5 + .5);
     
    
   
    
    // Mix the gradients using the Y value of the unit direction ray. 
    vec3 fog = mix(vec3(.5, .7, 1), vec3(.7, .6, .5), pow(max(rd.y*.7, 0.), 1.));
    col = mix(col, fog, smoothstep(0., .95, t/FAR)); // Linear fade. Much dustier. I kind of like it.
    //col = mix(col, fog, min(t*t*2./FAR/FAR, 1.)); // Quadratic fade off. More subtle.

    // Simulating mild sun scatter over scene: IQ uses it in his Elevated example.
    // The context in which I'm using it is not really physically correct, but it's only
    // mild, and I like the scattering effect.
    vec3 gLD = normalize(lp - ro);
    col += (dot(col, vec3(.299, .587, .114)) + .5)*sky.zyx*pow( max(dot(rd, gLD), 0.), 4.)*.2;
    
    
    // Adding in the glow.
    col += (col*.5 + .5)*vec3(.05, .2, 1)*min(glow, 1.);
   

    // Mild gamma correction before presenting to the screen.
    fragColor = vec4(sqrt(max(col, 0.)), 1);
    
}

//https://www.shadertoy.com/view/tsGXWm

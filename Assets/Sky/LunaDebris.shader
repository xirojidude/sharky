
Shader "Skybox/LunaDebris"
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

/*

    Lunar Debris
    ------------

    Just playing around a little more with 3D cellular tiling to create some random rocks floating
    around in space. The distance field is analogous to raymarched, mutated Voronoi. 

    Compared to genuine raymarched Voronoi, the framerate isn't too bad, but I think it can be
    improved upon. I'll take a closer look at it to see if I can increase it a bit.

    The trickiest thing to deal with was not the rocky field itself, but rather negotiating through
    it in a way that gives the appearance that the camera is naturally avoiding obstacles. I tried
    a few different things but smoothly negating space with a squashed diamond tube seemed to 
    produce the desired result. Anyway, the process is simple and is contained in the distance 
    function.


    Other examples:

    // Very stylish.
    Dusty Thing - kuvkar
    https://www.shadertoy.com/view/XlBXRt

    // Love the way this is done.
    Moon Rock - foxes
    https://www.shadertoy.com/view/4s33RX


*/

#define PI 3.14159265
#define FAR 60.

// Uses smooth combinations (smin, smax, etc) to smooth the rock joins.
#define SMOOTHING 

// Rotation matrix.
const float2x2 rM = float2x2(.7071, .7071, -.7071, .7071); 

// 2x2 matrix rotation. Note the absence of "cos." It's there, but in disguise, and comes courtesy
// of Fabrice Neyret's "ouside the box" thinking. :)
float2x2 rot2( float a ){ float2 v = sin(float2(1.570796, 0) + a);  return float2x2(v, -v.y, v.x); }

// Tri-Planar blending function. Based on an old Nvidia tutorial.
float3 tex3D( sampler2D tex, in float3 p, in float3 n ){
  
    n = max(abs(n), 0.001);//n = max((abs(n) - 0.2)*7., 0.001); //  etc.
    n /= (n.x + n.y + n.z ); 
    p = (tex2D(tex, p.yz)*n.x + tex2D(tex, p.zx)*n.y + tex2D(tex, p.xy)*n.z).xyz;
    return p*p;
}

// Smooth maximum, based on IQ's smooth minimum.
float smaxP(float a, float b, float s){
    
    float h = clamp( 0.5 + 0.5*(a-b)/s, 0., 1.);
    return lerp(b, a, h) + h*(1.0-h)*s;
}

// IQ's smooth minium function. 
float2 sminP(float2 a, float2 b , float s){
    
    float2 h = clamp( 0.5 + 0.5*(b-a)/s, 0. , 1.);
    return lerp(b, a, h) - h*(1.0-h)*s;
}

// IQ's smooth minium function. 
float sminP(float a, float b , float s){
    
    float h = clamp( 0.5 + 0.5*(b-a)/s, 0. , 1.);
    return lerp(b, a, h) - h*(1.0-h)*s;
}

// Cellular tile setup. Draw four overlapping objects (spheres, in this case) 
// at various positions throughout the tile.
 
float drawObject(in float3 p){
  
    p = frac(p)-.5;
    return dot(p, p);
    
}

   
float cellTile(in float3 p){

   
    float4 v, d; 
    
    d.x = drawObject(p - float3(.81, .62, .53));
    p.xy = mul(p.xy,rM);
    d.y = drawObject(p - float3(.6, .82, .64));
    p.yz = mul(p.yz,rM);
    d.z = drawObject(p - float3(.51, .06, .70));
    p.zx = mul(p.zx,rM);
    d.w = drawObject(p - float3(.12, .62, .64));

    // Obtaining the minimum distance.
    #ifdef SMOOTHING
    v.xy = sminP(d.xz, d.yw, .05); 
    #else
    v.xy = min(d.xz, d.yw);
    #endif
    
    // Normalize... roughly. Trying to avoid another min call (min(d.x*A, 1.)).
    #ifdef SMOOTHING
    return  sminP(v.x, v.y, .05)*2.5; 
    #else
    return  min(v.x, v.y)*2.5; 
    #endif    
    
}


// The path is a 2D sinusoid that varies over time, depending upon the frequencies, and amplitudes.
float2 path(in float z){ 
    
   
    //return float2(0); // Straight.
    float a = sin(z * 0.11);
    float b = cos(z * 0.14);
    return float2(a*4. -b*1.5, b*1.7 + a*1.5); 
    //return float2(a*4. -b*1.5, 0.); // Just X.
    //return float2(0, b*1.7 + a*1.5); // Just Y.
}


// Compact, self-contained version of IQ's 3D value noise function.
float n3D(float3 p){
    const float3 s = float3(7, 157, 113);
    float3 ip = floor(p); p -= ip; 
    float4 h = float4(0., s.yz, s.y + s.z) + dot(ip, s);
    p = p*p*(3. - 2.*p); //p *= p*p*(p*(p * 6. - 15.) + 10.);
    h = lerp(frac(sin(h)*43758.5453), frac(sin(h + s.x)*43758.5453), p.x);
    h.xy = lerp(h.xz, h.yw, p.y);
    return lerp(h.x, h.y, p.z); // Range: [0, 1].
}

// The debris field. I'll tidy it up later. In general, this is a terrible 
// distance field to hone in on. I'm come back to it later and rework it.
float map(float3 p){
    
    // Warping the whole field around the path.
    p.xy -= path(p.z);
    
    p/=2.;
    
    // Mutated, first order cellular object... the rocks.
    float3 q = p + (cos(p*2.52 - sin(p.zxy*3.5)))*.2;
    float sf = max(cellTile(q/5.), 0.); 
    
    // Mutated squashed diamond tube. Used to run the camera through.
    p += (cos(p*.945 + sin(p.zxy*2.625)))*.2;
    #ifdef SMOOTHING
    float t = .1 - abs(p.x*.05) - abs(p.y);
    #else
    float t = .05 - abs(p.x*.05) - abs(p.y);
    #endif  
    
    // Smoothly combine the negative tube space with the rocky field.
    //p = sin(p*4.+cos(p.yzx*4.));
    float n = smaxP(t, (.68 - (1.-sqrt(sf)))*2., 1.);// + abs(p.x*p.y*p.z)*.05;
   
    // A bit hacky... OK, very hacky. :)
    return n*3.;
    
}



// Surface bump function. I'm reusing the "cellTile" function, but absoulte sinusoidals
// would do a decent job too.
float bumpSurf3D( in float3 p, in float3 n){
    
    return (cellTile(p/2.))*.8 + (cellTile(p*1.5))*.2;
    
}

// Standard function-based bump mapping function.
float3 doBumpMap(in float3 p, in float3 nor, float bumpfactor){
    
    const float2 e = float2(0.001, 0);
    float ref = bumpSurf3D(p, nor);                 
    float3 grad = (float3(bumpSurf3D(p - e.xyy, nor),
                      bumpSurf3D(p - e.yxy, nor),
                      bumpSurf3D(p - e.yyx, nor) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
    
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


// Basic raymarcher.
float trace(in float3 ro, in float3 rd){

    float t = 0.0, h;
    for(int i = 0; i < 128; i++){
    
        h = map(ro+rd*t);
        // Note the "t*b + a" addition. Basically, we're putting less emphasis on accuracy, as
        // "t" increases. It's a cheap trick that works in most situations... Not all, though.
        if(abs(h)<0.0025*(t*.125 + 1.) || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.)
        t += h*.8;
        
    }

    return min(t, FAR);
    
}

// Ambient occlusion, for that self shadowed look. Based on the original by XT95. I love this 
// function, and in many cases, it gives really, really nice results. For a better version, and 
// usage, refer to XT95's examples below:
//
// Hemispherical SDF AO - https://www.shadertoy.com/view/4sdGWN
// Alien Cocoons - https://www.shadertoy.com/view/MsdGz2
float calculateAO( in float3 p, in float3 n )
{
    float ao = 0.0, l;
    const float maxDist = 2.;
    const float nbIte = 6.0;
    //const float falloff = 0.9;
    for( float i=1.; i< nbIte+.5; i++ ){
    
        l = (i*.75 + frac(cos(i)*45758.5453)*.25)/nbIte*maxDist;
        
        ao += (l - map( p + n*l ))/(1.+ l);// / pow(1.+l, falloff);
    }
    
    return clamp(1.- ao/nbIte, 0., 1.);
}


// Tetrahedral normal, to save a couple of "map" calls. Courtesy of IQ. In instances where there's no descernible 
// aesthetic difference between it and the six tap version, it's worth using.
float3 calcNormal(in float3 p){

    // Note the slightly increased sampling distance, to alleviate artifacts due to hit point inaccuracies.
    float2 e = float2(0.0025, -0.0025); 
    return normalize(e.xyy * map(p + e.xyy) + e.yyx * map(p + e.yyx) + e.yxy * map(p + e.yxy) + e.xxx * map(p + e.xxx));
}

/*
// Standard normal function. 6 taps.
float3 calcNormal(in float3 p) {
    const float2 e = float2(0.005, 0);
    return normalize(float3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
}
*/

// Shadows.
float shadows(in float3 ro, in float3 rd, in float start, in float end, in float k){

    float shade = 1.0;
    const int shadIter = 24; 

    float dist = start;
    //float stepDist = end/float(shadIter);

    for (int i=0; i<shadIter; i++){
        float h = map(ro + rd*dist);
        shade = min(shade, k*h/dist);
        //shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.

        dist += clamp(h, 0.02, 0.2);
        
        // There's some accuracy loss involved, but early exits from accumulative distance function can help.
        if ((h)<0.001 || dist > end) break; 
    }
    
    return min(max(shade, 0.) + 0.2, 1.0); 
}



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    
    // Screen coordinates.
//    float2 uv = (fragCoord - iResolution.xy*0.5)/iResolution.y;
    
    // Camera Setup.
    float3 lookAt = rd; //float3(0, 0, iTime*8. + 0.1);  // "Look At" position.
    float3 camPos = ro; //lookAt + float3(0.0, 0.0, -0.1); // Camera position, doubling as the ray origin.

 
    // Light positioning. The positioning is fake. Obviously, the light source would be much 
    // further away, so illumination would be relatively constant and the shadows more static.
    // That's what direct lights are for, but sometimes it's nice to get a bit of a point light 
    // effect... but don't move it too close, or your mind will start getting suspicious. :)
    float3 lightPos = camPos + float3(0, 7, 35.);

    // Using the Z-value to perturb the XY-plane.
    // Sending the camera, "look at," and two light floattors down the tunnel. The "path" function is 
    // synchronized with the distance function. Change to "path2" to traverse the other tunnel.
    lookAt.xy += path(lookAt.z);
    camPos.xy += path(camPos.z);
    //lightPos.xy += path(lightPos.z);

    // Using the above to produce the unit ray-direction floattor.
    float FOV = PI/3.; // FOV - Field of view.
    float3 forward = normalize(lookAt-camPos);
    float3 right = normalize(float3(forward.z, 0., -forward.x )); 
    float3 up = cross(forward, right);

    // rd - Ray direction.
    //float3 rd = normalize(forward + FOV*uv.x*right + FOV*uv.y*up);
    
    // Lens distortion.
    //float3 rd = (forward + FOV*uv.x*right + FOV*uv.y*up);
    //rd = normalize(float3(rd.xy, rd.z - length(rd.xy)*.25));    
    
    // Swiveling the camera about the XY-plane (from left to right) when turning corners.
    // Naturally, it's synchronized with the path in some kind of way.
    rd.xy = mul(rot2( path(lookAt.z).x/16. ),rd.xy);

    /*    
    // Mouse controls, as per TambakoJaguar's suggestion.
    // Works better if the line above is commented out.   
    float2 ms = float2(0);
    if (iMouse.z > 1.0) ms = (2.*iMouse.xy - iResolution.xy)/iResolution.xy;
    float2 a = sin(float2(1.5707963, 0) - ms.x); 
    float2x2 rM = float2x2(a, -a.y, a.x);
    rd.xz = rd.xz*rM; 
    a = sin(float2(1.5707963, 0) - ms.y); 
    rM = float2x2(a, -a.y, a.x);
    rd.yz = rd.yz*rM;
    */ 
    
    // Standard ray marching routine. I find that some system setups don't like anything other than
    // a "break" statement (by itself) to exit. 
    float t = trace(camPos, rd);    

    
    // Initialize the scene color.
    float3 sceneCol = float3(0,0,0);
    
    // The ray has effectively hit the surface, so light it up.
    if(t<FAR){
    
    
        // Surface position and surface normal.
        float3 sp = camPos + rd*t;
        
        // Voxel normal.
        //float3 sn = -(mask * sign( rd ));
        float3 sn = calcNormal(sp);
        
        // Sometimes, it's necessary to save a copy of the unbumped normal.
        float3 snNoBump = sn;
        
        // I try to avoid it, but it's possible to do a tex2D bump and a function-based
        // bump in succession. It's also possible to roll them into one, but I wanted
        // the separation... Can't remember why, but it's more readable anyway.
        //
        // tex2D scale factor.
        const float tSize0 = 1./2.;
        // tex2D-based bump mapping.
        sn = doBumpMap(_MainTex, sp*tSize0, sn, 0.1);
        float3 tsp =  sp;// + float3(0, 0, iTime/8.);// + float3(path(sp.z), 0.)

        // Function based bump mapping. Comment it out to see the under layer. It's pretty
        // comparable to regular beveled Voronoi... Close enough, anyway.
        sn = doBumpMap(tsp, sn, .5);
        
       
        // Ambient occlusion.
        float ao = calculateAO(sp, sn);//*.75 + .25;

        
        // Light direction floattors.
        float3 ld = lightPos-sp;

        // Distance from respective lights to the surface point.
        float lDist = max(length(ld), 0.001);
        
        // Normalize the light direction floattors.
        ld /= lDist;
        
        
        // Light attenuation, based on the distances above.
        float atten = 1./(1. + lDist*.007); // + distlpsp*distlpsp*0.025
        
        // Ambient light, due to light bouncing around the field, I guess.
        float ambience = 0.25;
        
        // Diffuse lighting.
        float diff = max( dot(sn, ld), 0.0);
    
        // Specular lighting.
        float spec = pow(max( dot( reflect(-ld, sn), -rd ), 0.0 ), 32.);

        
        // Fresnel term. Good for giving a surface a bit of a reflective glow.
        float fre = pow( clamp(dot(sn, rd) + 1., .0, 1.), 1.);

        // Object texturing, coloring and shading.
        float3 texCol = float3(.8, .9, 1.);
        texCol *= min(tex3D(_MainTex, sp*tSize0, sn)*3.5, 1.);
        texCol *= bumpSurf3D(sp, sn)*.5 + .5;
        
        // Shadows.
        float shading = shadows(sp + sn*.005, ld, .05, lDist, 8.);

        // Final color. Pretty simple.
        sceneCol = texCol*(diff + spec + ambience);
        
        // Adding a touch of Fresnel for a bit of space glow... I'm not so
        // sure if that's a thing, but it looks interesting. :)
        sceneCol += texCol*float3(.8, .95, 1)*pow(fre, 1.)*.5;


        // Shading.
        sceneCol *= atten*shading*ao;

        
       
    
    }
       
    // Blend in a bit of light fog for atmospheric effect. I really wanted to put a colorful, 
    // gradient blend here, but my mind wasn't buying it, so dull, blueish grey it is. :)
    float3 fog = float3(.6, .8, 1)/2.*(rd.y*.5 + .5);    
    sceneCol = lerp(sceneCol, fog, smoothstep(0., .95, t/FAR)); // exp(-.002*t*t), etc. fog.zxy

    // Clamp and present the badly gamma corrected pixel to the screen.
    fragColor = float4(sqrt(clamp(sceneCol, 0., 1.)), 1.0);
    
                return fragColor;
            }


            ENDCG
        }
    }
}

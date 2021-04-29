
Shader "Skybox/CellularLattice"
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

    Cellular Lattice
    ----------------

    Still playing around with 3D cellular tiles.

    Traversing a cellular, coral-like structure through an alien ocean ... Is anyone buying that?
    I don't know what it is either, but it looks grotesquely interesting. :)

    In technical terms, it's an intertwined sinusoidal lattice structure (a mutated gyroid of sorts) 
    with a prominent cellular surface attached.

    The scene was mildly inspired by IQ's Leizex demo and Tomasz Dobrowolski's Suboceanic.


    Cellular Tiled Tunnel 2 - Shane
    https://www.shadertoy.com/view/MdtSRl

    rgba leizex - Inigo Quilez
    http://www.pouet.net/prod.php?which=51829
    https://www.youtube.com/watch?v=eJBGj8ggCXU
    http://www.iquilezles.org/prods/index.htm

    Tomasz Dobrowolski - Suboceanic
    http://www.pouet.net/prod.php?which=18343

*/

#define FAR 20.

// Hash.
float hash( float n ){ return fract(cos(n)*45758.5453); }

// 2x2 matrix rotation. Note the absence of "cos." It's there, but in disguise, and comes courtesy
// of Fabrice Neyret's "ouside the box" thinking. :)
mat2 rot2( float a ){ vec2 v = sin(vec2(1.570796, 0) + a);  return mat2(v, -v.y, v.x); }

vec3 camPath(in float t){ return vec3(sin(t * 0.45)*.75, cos(t * 0.75)*.75, t); }

// Tri-Planar blending function. Based on an old Nvidia writeup:
// GPU Gems 3 - Ryan Geiss: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch01.html
vec3 tpl( sampler2D t, in vec3 p, in vec3 n ){
   
    n = max(abs(n), 0.001);
    n /= (n.x + n.y + n.z );  
    vec3 tx = (texture(t, p.yz)*n.x + texture(t, p.zx)*n.y + texture(t, p.xy)*n.z).xyz;
    
    return tx*tx;
}

// Akohdr's multitextured suggestion, with some small changes.
#define t2D texture
vec3 tpl4( sampler2D a, sampler2D b, 
           sampler2D c, in vec3 p, in vec3 n ){
    
    n = max(abs(n), 0.001);
    n /= (n.x + n.y + n.z );
    
    float h = dot(cos(p*6.), sin(p.yzx*6.));
    
    vec3 tx  = (t2D(a, p.yz)*n.x + t2D(a, p.zx)*n.y + t2D(a, p.xy)*n.z).xyz; // Pink sandstone.
    vec3 tx2 = (t2D(b, p.yz)*n.x + t2D(b, p.zx)*n.y + t2D(b, p.xy)*n.z).xyz; // Sandstone.
    vec3 tx3 = 1.-(t2D(c, p.yz)*n.x + t2D(c, p.zx)*n.y + t2D(c, p.xy)*n.z).zyx; // Pink coral.

    tx = mix(tx*tx, tx2*tx2, h*.5 + .5);
    
    h = dot(sin(p*5.), cos(p.zxy*5.));
    
    tx2 = mix(tx3*tx3, tx2*tx2, h*.5 + .5);
    
    return mix(tx, tx2, dot(sin(p*2.), sin(p.zxy*2.))*.5 + .5);
}

float drawObject(in vec3 p){
    
    // Anything that wraps the domain will work. The following looks pretty intereting.
    p = cos(p*6.2831853) + 1.;
    return dot(p, p);
    
}
/*
// Draw four warped spheres on a wrappable cube, and return the closest distance metric. Try to normalize
// the result between zero and one.
float cellTile(in vec3 p){
    
    vec4 d;
    
    // Draw four overlapping objects (spheres, in this case) at various positions throughout the tile.
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.39, .2, .11));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.62, .24, .06));
    p.xz = vec2(p.z-p.x, p.z + p.x)*.7071;
    d.w = drawObject(p - vec3(.2, .82, .64));

    d.xy = min(d.xy, d.zw); // Minimum distance determination.
    
    return 1.- min(d.x, d.y)*.166; // Normalize... roughly.
    
}
*/

// Fast, three tap version, but I feel four should be the minimum... Even so, this should give 
// a pretty convincing pattern, under the right circumstances.
float cellTile(in vec3 p){
    
    vec3 d;
    
    // Draw three overlapping objects (spherical, in this case) at various positions throughout the tile.
    d.x = drawObject(p - vec3(.81, .62, .53));
    p.xy = vec2(p.y-p.x, p.y + p.x)*.7071;
    d.y = drawObject(p - vec3(.2, .82, .64));
    p.yz = vec2(p.z-p.y, p.z + p.y)*.7071;
    d.z = drawObject(p - vec3(.41, .06, .70));
    
    return 1.- min(min(d.x, d.y), d.z)*.1666; // Normalize... roughly.
    
}



// A simple, cheap but visually effective sinusoid based lattice. The downside to building
// a scene with transcendentals is the honing difficulty.
float map(in vec3 p){
    
    float b = cellTile(p*3.); 
    //float b = cellTile(p + iTime/16.); // Animation.

    // Offsetting the lattice around the camera path.
    p.xy -= camPath(p.z).xy; 

    // Perturbing the surface slightly, prior to construction.
    p += (sin(p*3.14159 - sin(p.zyx*3.14159)*3.14159*1.25))*.1;

    // The main surface. A weird, molecular looking lattice.
    float n = abs(dot(cos(p*3.14159), sin(p.yzx*3.14159)));

    // Combining element to form the final structure.
    return .45 - n*.33  - b*.1;
    
}

 
// I keep a collection of occlusion routines... OK, that sounded really nerdy. :)
// Anyway, I like this one. I'm assuming it's based on IQ's original.
float cao(in vec3 pos, in vec3 nor)
{
    float sca = 1.5, occ = 0.0;
    for( int i=0; i<5; i++ ){
    
        float hr = 0.01 + float(i)*0.5/4.0;        
        float dd = map(nor * hr + pos);
        occ += (hr - dd)*sca;
        sca *= 0.7;
    }
    return clamp( 1.0 - occ, 0.0, 1.0 );    
}


// Tetrahedral normal, courtesy of IQ.
vec3 nr(in vec3 p)
{  
    vec2 e = vec2(-1, 1)*.001;   
    return normalize(e.yxx*map(p + e.yxx) + e.xxy*map(p + e.xxy) + 
                     e.xyx*map(p + e.xyx) + e.yyy*map(p + e.yyy) );   
}


// Basic raymarcher.
float trace(in vec3 ro, in vec3 rd){
    
    float t = 0., h;
    for(int i = 0; i < 128; i++){

        h = map(ro+rd*t);
        // Note the "t*b + a" addition. Basically, we're putting less emphasis on accuracy, as
        // "t" increases. It's a cheap trick that works in most situations... Not all, though.
        if(abs(h)<0.0015*(t*.125 + 1.) || t>FAR) break; // Alternative: 0.001*max(t*.25, 1.)
        t += (step(1., h)*.25 + .5)*h;
        
    }

    return min(t, FAR);
}

// Shadows.
float sha(in vec3 ro, in vec3 rd, in float start, in float end, in float k){

    float shade = 1.0;
    const int shadIter = 24; 

    float dist = start;
    //float stepDist = end/float(shadIter);

    for (int i=0; i<shadIter; i++){
        float h = map(ro + rd*dist);
        //shade = min(shade, k*h/dist);
        shade = min(shade, smoothstep(0.0, 1.0, k*h/dist)); // Subtle difference. Thanks to IQ for this tidbit.

        dist += clamp(h, 0.02, 0.16);
        
        // There's some accuracy loss involved, but early exits from accumulative distance function can help.
        if ((h)<0.001 || dist > end) break; 
    }
    
    return min(max(shade, 0.) + 0.3, 1.0); 
}


// Surface bump function. Cheap, but with decent visual impact.
float bumpSurf3D( in vec3 p){
    
    float vor = cellTile(p*27.);
    
    return pow(max(vor, 0.), 18.);

}

// Standard function-based bump mapping function.
vec3 dbF(in vec3 p, in vec3 nor, float bumpfactor){
    
    const vec2 e = vec2(0.001, 0);
    float ref = bumpSurf3D(p);                 
    vec3 grad = (vec3(bumpSurf3D(p - e.xyy),
                      bumpSurf3D(p - e.yxy),
                      bumpSurf3D(p - e.yyx) )-ref)/e.x;                     
          
    grad -= nor*dot(nor, grad);          
                      
    return normalize( nor + grad*bumpfactor );
    
}

// Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total.
vec3 db( sampler2D tx, in vec3 p, in vec3 n, float bf){
   
    const vec2 e = vec2(0.001, 0);
    
    // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.    
    mat3 m = mat3( tpl(tx, p - e.xyy, n), tpl(tx, p - e.yxy, n), tpl(tx, p - e.yyx, n));
    
    vec3 g = mul(vec3(0.299, 0.587, 0.114),m); // Converting to greyscale.
    g = (g - dot(tpl(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);
                      
    return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.01+ _XYZPos*100;                                             // ray origin

    
    
    // Screen coordinates.
    vec2 u = (fragCoord - iResolution.xy*0.5)/iResolution.y;
    
    // Perturbing the screen coordinates to create the lamest underwater effect ever. :)
    // Seriously, though, it's quite effective, all things considered.
    u += sin(u*32. + cos(u.yx*16. + iTime*4.))*.0035;

    
    // Camera Setup.
    vec3 o = ro; //camPath(iTime*.5); // Camera position, doubling as the ray origin.
    vec3 lk = camPath(iTime*.5 + .1);  // "Look At" position.
    vec3 l = camPath(iTime*.5 + 1.5) + vec3(.0, .0, 0.); // Light position, somewhere near the moving camera.


    // Using the above to produce the unit ray-direction vector.
    float FOV = 3.14159/3.; // FOV - Field of view.
    vec3 fwd = normalize(lk-o);
    vec3 rgt = normalize(vec3(fwd.z, 0., -fwd.x )); 
    vec3 up = cross(fwd, rgt);

    // Unit direction ray.
    vec3 r = rd; //normalize(fwd + FOV*(u.x*rgt + u.y*up));
    // Lens distortion.
    //vec3 r = fwd + FOV*(u.x*rgt + u.y*up);
    //r = normalize(vec3(r.xy, (r.z - length(r.xy)*.5)));
    
    // Swiveling the camera from left to right when turning corners.
//    r.xy = mul(rot2( camPath(lk.z).x/16. ),r.xy);
 
    // Raymarch.
    float t = trace(o, r);

    // Initialize the scene color to zero.
    vec3 col = vec3(0,0,0);
    
    // If the surface is hit, light it up.
    if(t<FAR){
    
        // Position and normal.
        vec3 p = o + r*t, n = nr(p);
        
        // Texture bump the normal.
        float sz = 1./1.;
        n = db(_MainTex, p*sz, n, .02/(1. + t/FAR));
        
        n = dbF(p*sz, n, .01);


        l -= p; // Light to surface vector. Ie: Light direction vector.
        float d = max(length(l), 0.001); // Light to surface distance.
        l /= d; // Normalizing the light direction vector.
        
        // Ambient occlusion and shadowing.
        float ao =  cao(p, n);
        float sh = sha(p, l, 0.04, d, 4.);
        
        // Diffuse, specular, fresnel.
        float di = max(dot(l, n), 0.);
        float sp = pow(max( dot( reflect(r, n), l ), 0.0 ), 8.); // Specular term.
        float fr = clamp(1.0 + dot(r, n), 0.0, 1.0); // Fresnel reflection term.
        
        // Texturing the surface with some tri-planar mapping..
        vec3 tx = tpl(_MainTex, p*sz, n);
        
        // Texture variance: Akohdr's suggestion.
        // Requires an additional sandstone texture in iChannel1, and the pink coral texture
        // in iChannel2.
        //vec3 tx = tpl4(_MainTex, _MainTex1, _MainTex2, p*sz, n);
        
        float c = dot(tx, vec3(0.299, 0.587, 0.114));
        
        tx += vec3(c*c*.8, c, c*c*.5)*fr;
        

        // Very simple coloring. Fresnel and texture combination.
        col = tx*(di + .1 + sp)+ tx*fr*2.;
        col *= 1./(1. + d*.125 + d*d*.025)*ao*sh;

        
    }

    // Mixing in a simple blue background.
    vec3 bg = vec3(.5, .7, 1);
    col = mix(clamp(col, 0., 1.), bg, smoothstep(0., FAR-5., t));
    
    // Half hearted gamma correction.
    fragColor = vec4(sqrt(clamp(col, 0., 1.)), 1.);


                return fragColor;
            }

            ENDCG
        }
    }
}




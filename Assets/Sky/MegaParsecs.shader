
Shader "Skybox/MegaParsecs"
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


// "Megaparsecs" by Martijn Steinrucken aka BigWings/CountFrolic - 2020
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Email: countfrolic@gmail.com
// Twitter: @The_ArtOfCode
// YouTube: youtube.com/TheArtOfCodeIsCool
//
// Music:
// https://soundcloud.com/astropilot/space-ambient-demo-00
//
// My attempt at a spiral galaxy. Made from a bunch of ellipses.
//
// I got the idea when I read about new theories concerning spiral
// galaxy formation. The standard idea of the spiral arms just orbiting
// around the center doesn't work because it would 'wind up' the spiral
// with every rotation. This would very quickly (on cosmological scales) 
// lead the spirals to would wind up to the point where they'd dissapear. 
// This is not what we observe so a different theory is needed. 
// One such theory is that we are actually looking at lots of
// overlapping and rotated elliptical orbits, which I tried to emulate here.
// 
// Playing around with this made me realize that a galaxy is just a big
// vortex of dust, where every dust particle is a star!
// I threw in some super big & bright stars to make for a more interesting
// image. There is also the occasional supernova thrown in. Though in
// reality super novae happen in our galaxy about once every 50 years and
// since one orbit takes about 250 million years it should have about 
// 150,000 super nova flashes per second if we look at it at this
// time-scale!
//
// I am not the first one to render a galaxy as a series of eliptical
// orbits. As with many other topics, Fabrice Neyret beat me to the punch:
// https://www.shadertoy.com/view/Ms3czX

#define S smoothstep
#define T iTime

#if HW_PERFORMANCE==0
#define NUMRINGS 20.
#define MAX_BLOCKS 20.
#else
#define NUMRINGS 40.
#define MAX_BLOCKS 40.
#endif

mat2 Rot(float a) {
    float s=sin(a),c=cos(a);
    return mat2(c,-s,s,c);
}

float Hash31(vec3 p3) {
    p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 GetRayDir(vec2 uv, vec3 p, vec3 l, vec3 up, float z) {
    vec3 f = normalize(l-p),
        r = normalize(cross(up, f)),
        u = cross(f,r),
        c = f*z,
        i = c + uv.x*r + uv.y*u,
        d = normalize(i);
    return d;
}

vec4 Galaxy(vec3 ro, vec3 rd, float seed, float a1, float a2, float cut) {
    
    mat2 rot = Rot(a1);
    ro.xy = mul(ro.xy,rot); 
    rd.xy= mul(rd.xy,rot);
    rot=Rot(a2);
    ro.yz = mul(ro.yz,rot); 
    rd.yz= mul(rd.yz,rot);
    
    vec2 uv = ro.xz+(ro.y/-rd.y)*rd.xz;
 
    seed = fract(sin(seed*123.42)*564.32);
        
    vec3 
        col = vec3(0,0,0),
        dustCol = vec3(.3, .6, 1.);
    
    float alpha = 0.;
    if(cut==0. || (ro.y*rd.y<0. && length(uv)<2.5)) {
        
        float 
            ringWidth = mix(10.,25., seed),
            twist = mix(.3, 2., fract(seed*10.)),
            numStars = mix(2., 15., pow(fract(seed*65.),2.)),
            contrast = fract(seed*3.),
            flip = 1.,
            t=T*.1*sign(seed-.5),
            z, r, ell, n, d, sL, sN, i;
        
        if(cut==0.) twist = 1.;
        
        for(i=0.; i<1.; i+=1./NUMRINGS) {

            flip *= -1.;
            z = mix(.06, 0., i)*flip*fract(sin(i*563.2)*673.2);
            r = mix(.1, 1., i);

            uv = ro.xz+((ro.y+z)/-rd.y)*rd.xz;
        
            vec2 st = mul(uv,Rot(i*6.2832*twist));
            st.x *= mix(2., 1., i);

            ell = exp(-.5*abs(dot(st,st)-r)*ringWidth);
            vec2 texUv = mul(.2*st,Rot(i*100.+t/r));
            vec3 
                dust = texture(_MainTex, texUv+i).rgb,
                dL = pow(ell*dust/r, vec3(.5+contrast,.5+contrast,.5+contrast));

            vec2 id = floor(texUv*numStars);
            texUv= fract(texUv*numStars)-.5;

            n = Hash31(id.xyy+i);

            d = length(texUv); 

            sL = S(.5, .0, d)*pow(dL.r,2.)*.2/d;
           
            sN = sL;
            sL *= sin(n*784.+T)*.5+.5;
            sL += sN*S(.9999,1., sin(n*784.+T*.05))*10.;
            col += dL*dustCol;

            alpha += dL.r*dL.g*dL.b;

            if(i>3./numStars)
            col += sL* mix(vec3(.5+sin(n*100.)*.5, .5, 1.), vec3(1,1,1), n);
        }

        col = col/NUMRINGS;
    }
    
    float ex = exp(-.5*dot(uv,uv)*30.);
    vec3 
        tint = 1.-vec3(pow(seed,3.), pow(fract(seed*98.),3.), 0.)*.5,
        center = vec3( ex,ex,ex ),
        cp = ro + max(0., dot(-ro, rd))*rd;
    
    col *= tint;
    
    cp.y*= 4.;
    center += dot(rd, vec3(rd.x, 0, rd.z))*exp(-.5*dot(cp,cp)*50.);
    
    col += center*vec3(1., .8, .7)*1.5*tint;
    
    return vec4(col, alpha);
}

vec3 Bg(vec3 rd) {
    vec2 uv = vec2(atan(rd.x,rd.z), rd.y*.5+.5);
    uv *= 2.;
    float wave = sin(rd.y*3.14+T*.1)*.5+.5;
    wave *= sin(uv.x+uv.y*3.1415)*.5+.5;
    return vec3(0.01*sin(T*.06),0,.05)*wave;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

   vec2 
        uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
//        M = iMouse.xy/iResolution.xy;
    
    float 
        t = T*.1,
        dolly = (1.-sin(t)*.6),
        zoom = mix(.3, 2., pow(sin(t*.1), 5.)*.5+.5),
        dO = 0.;
    
    //vec3 
    ro = ro + vec3(0,2,-2)*dolly;
//    ro.yz *= Rot(M.y*5.+sin(t*.5));
//    ro.xz *= Rot(-M.x*5.+t*.1);
    vec3 up = vec3(0,1,0);
    up.xy = mul(up.xy, Rot(sin(t*.2)));
    vec3 
        //rd = GetRayDir(uv, ro, vec3(0), up, zoom),
        col = Bg(rd),
        dir = sign(rd)*.5;
    [loop]
    for(float i=0.; i<MAX_BLOCKS; i++) {
        vec3 p = ro+dO*rd;
        
        p.x += T*.2;
        vec3 
            id = floor(p),
            q = fract(p)-.5,
            rC = (dir-q)/rd;    // ray to cell boundary
        
        float 
            dC = min(min(rC.x, rC.y), rC.z)+.0001,      // distance to cell just past boundary
            n = Hash31(id);
        
        dO += dC;
        
        if(n>.01) continue;
        
        float 
            a1 = fract(n*67.3)*6.2832,
            a2 = fract(n*653.2)*6.2832;
        
        col += Galaxy(q*4., rd, n*100., a1, a2,1.).rgb*S(25., 10., dO);
    }
    
    vec4 galaxy = Galaxy(ro, rd, 6., 0., 0.,0.);

    float 
        alpha = pow(min(1., galaxy.a*.6),1.),
        a = atan(uv.x,uv.y),
        sB = sin(a*13.-T)*sin(a*7.+T)*sin(a*10.-T)*sin(a*4.+T),
        d = length(uv);
    
    sB *= S(.0, .3, d);
    col = mix(col, galaxy.rgb*.1, alpha*.5);
    col += galaxy.rgb;
    col += max(0., sB)*S(5.0, 0., dot(ro,ro))*.03*zoom;
    
    col *= S(1., 0.5, d);
    
    fragColor = vec4(sqrt(col),1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}



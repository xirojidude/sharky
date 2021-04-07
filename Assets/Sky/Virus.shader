
Shader "Skybox/Virus"
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


// "Corona Virus" 
// by Martijn Steinrucken aka BigWings/CountFrolic - 2020
// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Music: Xerxes - early-morning-crystals
//
// This effect depicts my idea of what a virus could
// look like taking a huge artistic license. I started making what
// I imagine to be a lipid bilayer and then realized.. a virus
// doesn't have one! So then I just figured I'd make it look 'mean'
// 
// At first I tried using sphere coordinates but they distort too 
// much and have poles, so I ended up with projected cubemap
// coordinates instead. I think the function WorldToCube below
// is extremely usefull if you want to bend stuff around a sphere
// without too much distortion.
//
// As usual, the code could be a lot better and cleaner but I figure
// that by the time its all clean, I've lost interest and the world
// has moved on. Better ship it while its hot ;)
//
// uncomment the MODEL define to see once particle by itself
// you can change the amount of particles by changing the
// FILLED_CELLS define


//#define MODEL
#define FILLED_CELLS .3

#define PI 3.1415
#define TAU 6.2831

#define MAX_STEPS 400
#define MAX_DIST 40.
#define SURF_DIST .01



float2x2 Rot(float a) {
    float s = sin(a);
    float c = cos(a);
    return float2x2(c, -s, s, c);
}

float smin( float a, float b, float k ) {
    float h = clamp( 0.5+0.5*(b-a)/k, 0., 1. );
    return lerp( b, a, h ) - k*h*(1.0-h);
}

float sdCapsule(float3 p, float3 a, float3 b, float r) {
    float3 ab = b-a;
    float3 ap = p-a;
    
    float t = dot(ab, ap) / dot(ab, ab);
    t = clamp(t, 0., 1.);
    
    float3 c = a + t*ab;
    
    return length(p-c)-r;
}

float N31(float3 p) {
    float3 a = frac(float3(p) * float3(213.897, 653.453, 253.098));
    a += dot(a, a + 79.76);
    return frac(a.x * a.y * a.z);
}

float N21(float2 p) {
    float3 a = frac(float3(p.xyx) * float3(213.897, 653.453, 253.098));
    a += dot(a, a.yzx + 79.76);
    return frac((a.x + a.y) * a.z);
}

float3 SphereCoord(float3 p) {
    float x = atan2( p.z,p.x);
    float y = atan2( p.y,length(p.xz));
    
    return float3(x/TAU, length(p), 2.*y/TAU);
}

// returns cubemap coordinates
// xy = uv coords for face of cube, z = cube index (-3,-2,-1, 1, 2, 3)
float3 WorldToCube(float3 p) {
    float3 ap = abs(p);
    float3 sp = sign(p);
    float m = max(ap.x, max(ap.y, ap.z));
    float3 st;
    if(m==ap.x)
        st = float3(p.zy, 1.*sp.x);
    else if(m==ap.y)
        st = float3(p.zx, 2.*sp.y);
    else
        st = float3(p.xy, 3.*sp.z);
    
    st.xy /= m;
    
    // mattz' distortion correction
    st.xy *= (1.45109572583 - 0.451095725826*abs(st.xy));
       
    return st;
}

float Lipid(float3 p, float twist, float scale) {
    float3 n = sin(p*20.)*.2;
    p *= scale;
    
    p.xz=mul(p.xz , Rot(p.y*.3*twist));
    p.x = abs(p.x);
    
    float d = length(p+n)-2.;
    
    float y = p.y*.025;
    float r = .05*scale;
    float s = length(p.xz-float2(1.5,0))-r+max(.4,p.y);
    d = smin(d, s*.9,.4);
    
    return d/scale;
}

float sdTentacle(float3 p) {
    float offs = sin(p.x*50.)*sin(p.y*30.)*sin(p.z*20.);
    
    p.x += sin(p.y*10.+_Time.y)*.02;
    p.y *= .2;
    
    float d = sdCapsule(p, float3(0,0.1,0), float3(0,.8,0), .04);
    
    p.xz = abs(p.xz);
    
    d = min(d, sdCapsule(p, float3(0,.8,0), float3(.1,.9,.1), .01));
    d += offs*.01;
    
    return d;
}


float Particle(float3 p, float scale, float amount) {  
    float t = _Time.y;
 
    float3 st = WorldToCube(p);
    float3 cPos = float3(st.x, length(p), st.y);
    float3 tPos = cPos;
    
    float3 size = float3(.05,.05,.05);
    
    cPos.xz *= scale;
    float2 uv = frac(cPos.xz)-.5;
    float2 id = floor(cPos.xz);
    
    uv = frac(cPos.xz)-.5;
    id = floor(cPos.xz);
    
    
    float n = N21(id);
    
    t = (t+st.z+n*123.32)*1.3;
    float wobble = sin(t)+sin(1.3*t)*.4;
    wobble /= 1.4;
    
    wobble *= wobble*wobble;
    
    wobble = wobble*amount/scale;
    float3 ccPos = float3(uv.x, cPos.y, uv.y);
    float3 sPos = float3(0, 3.5+wobble, .0);
    
    float3 pos = ccPos-sPos;
    
    pos.y *= scale/2.;
   
    float r = 16./scale;
    r/=sPos.y; // account for height
    float d = length(pos)-r;
    d = Lipid(pos, n, 10.)/scale;
    
    d = min(d, length(p)-.2*scale); // inside blocker
    
    
    float tent = sdTentacle(tPos);
    d = min(d, tent);
    
    return d;
}

float dCell(float3 p, float size) {
    p = abs(p);
    float d = max(p.x, max(p.y, p.z));
    
    return max(0., size - d);
}

float GetDist(float3 p) {
    float t = _Time.y;
    
    float scale=8.;
    
    #ifndef MODEL
    p.z += t;
    float3 id = floor(p/10.);
    p = ((p)% float3(10,10,10))-5.;
    float n = N21(id.xz);
    p.xz = mul(p.xz,Rot(t*.2*(n-.5)));
    p.yz = mul(p.yz,Rot(t*.2*(N21(id.zx)-.5)));
    scale = lerp(4., 16., N21(id.xz));//mod(id.x+id.y+id.z, 8.)*2.;
    
    n = N31(id);
    if(n>FILLED_CELLS) {            // skip certain cells
        return dCell(p, 5.)+.1;
    }
    #endif
    
   
    
    p += sin(p.x+t)*.1+sin(p.y*p.z+t)*.05;
    
   
    float surf = sin(scale+t*.2)*.5+.5;
    surf *= surf;
    surf *= 4.;
    surf += 2.;
    float d = Particle(p, scale, surf);
    
    p.xz = mul(p.xz, Rot(.78+t*.08));
    p.zy = mul(p.zy,Rot(.5));
    
    d = smin(d, Particle(p, scale, surf), .02);
    
    
    return d;
}

float RayMarch(float3 ro, float3 rd) {
    float dO=0.;
    float cone = .0005;
    for(int i=0; i<MAX_STEPS; i++) {
        float3 p = ro + rd*dO;
        
        float dS = GetDist(p);
        dO += dS;
        if(dO>MAX_DIST || abs(dS)<SURF_DIST+dO*cone) break;
    }
    
    return dO;
}

float3 GetNormal(float3 p) {
    float d = GetDist(p);
    float2 e = float2(.001, 0);
    
    float3 n = d - float3(
        GetDist(p-e.xyy),
        GetDist(p-e.yxy),
        GetDist(p-e.yyx));
    
    return normalize(n);
}


float3 R(float2 uv, float3 p, float3 l, float3 up, float z) {
    float3 f = normalize(l-p),
        r = normalize(cross(up, f)),
        u = cross(f,r),
        c = p+f*z,
        i = c + uv.x*r + uv.y*u,
        d = normalize(i-p);
    return d;
}



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

//   float2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;
//    float2 m = (iMouse.xy-.5*iResolution.xy)/iResolution.xy;
//    if(m.x<-0.45&&m.y<-.45) m += .5;
    
    float t = _Time.y;
    
    float3 col = float3(0,0,0);
    
    #ifdef MODEL
//    float3 ro = float3(0, 0, -7);
    //ro.y += sin(t*.1)*3.;
    ro.yz *= Rot(-m.y*2.);
    ro.xz *= Rot(_Time.y*.0-m.x*6.2831);
//    float3 rd = R(uv, ro, float3(0,0,0), float3(0,1,0), .5);
    //ro += 3.;
#else
//    float3 ro = float3(0, 0, -1);
    //ro.y += sin(t*.1)*3.;
//    ro.yz *= Rot(-m.y*2.);
//    ro.xz *= Rot(_Time.y*.0-m.x*6.2831);
    
//    float3 up = float3(0,1,0);
//    up.xy *= Rot(sin(t*.1));
//    float3 rd = R(uv, ro, float3(0,0,0), up, .5);
    
//    ro.x += 5.;
//    ro.xy *= Rot(t*.1);
//    ro.xy -= 5.;
    #endif
    
    float d = RayMarch(ro, rd);
    
    float bg = rd.y*.5+.3;
    float poleDist = length(rd.xz);
    float poleMask = smoothstep(.5, 0., poleDist);
    bg += sign(rd.y)*poleMask;
    
    float a = atan2( rd.z,rd.x);
    bg += (sin(a*5.+t+rd.y*2.)+sin(a*7.-t+rd.y*2.))*.2;
    float rays = (sin(a*5.+t*2.+rd.y*2.)*sin(a*37.-t+rd.y*2.))*.5+.5;
    bg *= lerp(1., rays, .25*poleDist*(sin(t*.1)*.5+.5));//*poleDist*poleDist*.25;
    col += bg;
    
    if(d<MAX_DIST) {
        float3 p = ro + rd * d;
   
        float3 n = GetNormal(p);
       
        #ifndef MODEL
        p = ((p)% float3(10,10,10))-5.;
        #endif
        
        float s = dot(n, normalize(p))-.4;
        float f = -dot(rd, n);
        
        col += dot(n,-rd)*.5+.5;
        //col += (1.-f*f)*s*1.5;
        
        col *= 0.;
        float r = 3.7;
        float ao = smoothstep(r*.8, r, length(p));
        col += (n.y*.5+.5)*ao*2.;
        //col *= 2.;
        col *= smoothstep(-1., 6., p.y);
        
        //col += n*.5+.5;
    }
    
    col = lerp(col, float3(bg,bg,bg), smoothstep(0., 40., d));
    
    //col *= float3(1., .9, .8);
    //col = 1.-col;
    fragColor = float4(col,1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}


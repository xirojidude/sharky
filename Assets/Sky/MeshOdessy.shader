
Shader "Skybox/MeshOdessy"
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
            #define dFdx ddx
            #define dFdy ddy
            #define iChannel0 _MainTex

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


//        ___.                       ___            .___ ___________
// ,___ _\(   |._____________ __. __\(__/___. ______\|_  \\          )_. ________,
// |%%% |     ||            //  |/          |/        /                | %%%%%%%%|
// |%%% |    _||                |           |        _                 | %%%%%%%%|
// |'_ _|    \                 _|          _|        |                 |_ _ `%%%%|
//  .\)\_______________________)|__________)|________|__________________/(/.
// <------------------------------------------------------------------------diP->>
//         l a t i t u d e  i n d e p e n d e n t  a s s o c i a t i o n
// <<---------------------------------------------------------------------------->
// 
//                                     L.I.A.
// 
//                                    presents
// 
//                                 "Mesh Odyssey"
// 
// 
//                             a 4 kilobytes PC intro
// 
// 
// 
// --------------------------------- Release Info --------------------------------
// 
// 
//                                  code by: Kali
//                                  music by: Uctumi
// 
//    We bring you this time "Mesh Odyssey", our latest 4kb intro for Revision
//    2021 online. Tools used: Leviathan 2.0 / 4klang 3.0
//    This one's a little encoder killer, so it's better to run it on a nice
//    PC in order to watch it properly.
// 
//    Kali wants to thank Barbi, Alejo, Migue and Keyla for their support.
// 
// 
// ------------------------------------ Members -----------------------------------
// 
// 
// 
// 
// 
//             Bitnenfer - Foco - Kali - Pixel Syndrome - Riq - Uctumi
// 
// 
// ----------------------------------- Contact -----------------------------------
// 
// 
//                          http://liagroup.is-great.org
// 
// 
// ---------------------------------- Greetings ----------------------------------
// 
// 
//                      To all fellow demosceners of the world!
//             Specially: PVM - Hokuto Force - Azop - Christian Kleinhus
//                           Dipswitch - QOP - JPupper
// 
// 
// -------------------------------------------------------------------------------
// 
// 
// 
//                         Signing off... The LIA crusaders


#define rot(a) mat2(cos(a),sin(a),-sin(a),cos(a))
#define sm(a,b,c) smoothstep(a,b,c)
//#define iTime iChannelTime[0]


float it=0., gcol=0.,hcol=0.,tcol=0.,fcol=0.,shcol=0., t, det=.0015, turbo;
vec3 shdir,shfrom;


float hash12(vec2 p)
{
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float dot2( in vec3 v ) { return dot(v,v); }
float sdBoundingBox( vec3 p, vec3 b, float e)
{
       p = abs(p  )-b;
  vec3 q = abs(p+e)-e;

  return sqrt(min(min(dot2(max(vec3(p.x,q.y,q.z),0.0)),
                      dot2(max(vec3(q.x,p.y,q.z),0.0))),
                      dot2(max(vec3(q.x,q.y,p.z),0.0)))) 
         +min(0.0,min(min( max(p.x,max(q.y,q.z)),
                           max(p.y,max(q.z,q.x))),
                           max(p.z,max(q.x,q.y))));
}

float line1( vec2 p, vec2 a, vec2 b)
{
  vec2 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h );
}

float logo(vec2 uv) {
    uv.y+=3.4;
    uv*=.4;
    uv.y*=-1.2;
    float c = line1(uv, vec2(-.65,.3), vec2(-.65,-.3));    
    c = min(c,line1(uv, vec2(-.65,-.3), vec2(-.35,-.3)));
    c = min(c,line1(uv, vec2(-.15,.3), vec2(-.15,-.3)));
    c = min(c,line1(uv, vec2(.25,.3), vec2(.05,-.3)));
    c = min(c,line1(uv, vec2(.3,.3), vec2(.50,-.3)));
    c = min(c,line1(uv, vec2(.5,-.3), vec2(.2,-.3)));
    return sm(.06,.05,c);
}


mat3 lookat(vec3 dir, vec3 up) {
    vec3 r=cross(dir,up);
    return mat3(r,cross(dir,r),dir);
}


vec3 path(float t) {
    return vec3(sin(t*.25)*1.5,cos(t*.5)+3.+sm(0.,2.,1.+sin(t*.25))*2.,t);
}

vec3 pathcam(float t) {
    vec3 p=path(t);
    p.x+=sm(0.,2.,1.+sin(t*.5))*(1.+sin(t*.09)) * sm(170.,150.,t);
    p.y-=.2-sm(55.,70.,t) * 2.5 + (sm(90.,100.,t)-sm(120.,130.,t))*1.5;
    return p;
}

float ship(vec3 p)
{
    float bou=length(p)-.25, ale, tur, cas;
    if (bou>det+.1) return bou;
    p=mul(lookat(shdir,vec3(0.,1.,0.)),p);
    p.z*=-1.;
    p.xy=mul(p.xy,rot(shdir.x*1.7-sm(80.,81.,iTime)*3.1416*2.));
    vec3 p2 = p;
    p.z*=.9;
    p.y*=5.+sm(.05,.1,abs(p.x))*4.;
    p.x*=1.+sm(0.1,0.,p.z);
    p.y-=sm(.07,.05,abs(p.x))*.15*sm(0.,-.1,p.z)*sm(-.2,-.1,p.z)*step(0.,p.y)+sm(.05,.1,abs(p.x))*.2*sm(-.15,.1,p.z);
    p.z-=sm(.1,.22,abs(p.x))*.1;
    cas=max(length(p)-.2,p.z-.18);
    p=p2;
    p.y-=.025;
    p.z-=.1;
    p.yz=mul(p.yz,rot(-.3));
    ale = length(max(vec3(0.,0,0),abs(p)-vec3(-.00,.03,.09)))-.004;
    cas=min(cas,ale*2.);
    p=p2;
    p.z-=.2;
    p.z*=.15-step(0.1,turbo)*.1;
    p.x=abs(p.x)-.025;
    tur=length(p)-.001+sin(iTime*30.+sign(p2.x))*.001;
    shcol=.3+sm(.02,.019,abs(p.x-.01))*.4;
    if (p2.y>0.01&&p2.z<-.01&&abs(p2.x)<.03) shcol=0.01;
    if (p2.z>0.18&&abs(p2.x)<.05&&p.y<.03) shcol=0.0;
    tcol=max(tcol,max(0.,.02-tur)/.02);
    return cas*.4;
}

float de(vec3 pos) {
    shcol=0.;
    float tur=length(pos.xy-pathcam(pos.z).xy)-.3,
        shi=ship(pos-shfrom),
        sph=length(pos-path(-8.)+vec3(0.,3.,0.))-10.-abs(.5-fract(pos.y*5.))*.1*sm(5.,3.,pos.y),
        ch=sm(80.,81.,iTime),
        z = pos.z, d=1000.,o=d,der=1.,sc, l;
    pos.xy-=path(pos.z).xy;
    if (iTime<12.) l=logo(pos.xy);
    mat2 rot1 = rot(z*.05*(1.-ch)+sm(94.5,95.5,iTime)*1.5), rot2 = rot(-ch);
    pos.y-=1.2;
    pos.x-=1.;
    pos.xy=mod(pos.xy+4.5,9.)-4.5;
    pos.z=mod(pos.z,8.)-4.;
    vec3 p=pos;
    for (int i=0; i<7; i++) {
        p.xy=mul(p.xy,rot1);
        p.yz=mul(p.yz,rot2);
        p.xz=abs(p.xz);
        p.y+=1.;
        sc=1.5/clamp(dot(p,p),0.1,1.0);
        p=p*sc-vec3(2.,1.,3.);
        der*=sc;
        float shp=sdBoundingBox(p,vec3(1.,1.,2.),.02)/der;
        if (shp<d && i>1) {
            d=shp;
            it=float(i);
        }
        o=min(o,dot(p.xy,p.xy));
    }
    d=min(d,length(p.xy)/der-.01);
    d-=.003*ch;
    gcol=step(fract(iTime*1.17+it/6.),.15)*6.*(step(26.5,iTime)-step(106.5,iTime));
    hcol+=pow(fract(pos.z*.2+iTime*.7+p.z*.02),50.)*.2*step(iTime,106.5);
    fcol=exp(-10.*o)*.7;
    d=min(d,sph);
    d=max(d,-tur);
    d=min(d,shi);
    if (d==sph) gcol=5.+l*5.; fcol+=l*.5;
    if (d==shi) gcol=fcol=hcol=0.; else shcol=0.;
    hcol+=sm(123.,130.,iTime);
    return d*.85;
}

vec3 normal(vec3 p) {
    vec2 e=vec2(0.,det);
    return normalize(vec3(de(p+e.yxx),de(p+e.xyx),de(p+e.xxy))-de(p));
}

vec3 shade(vec3 p, vec3 dir, vec3 n) {
    return (vec3(.025,.075,.15)+shcol+fcol)*abs(n.x);
}


vec3 march(vec3 from, vec3 dir) {
    vec3 p=from, col=vec3(0.,0,0), refcol=col;
    float totdist=0.,d=0.,ref=0.,g=0.,h=hash12(dir.xy*1000.+iTime);
    for(int i=0; i<100; i++) {
        p+=dir*d*(.9+h*.2);
        d=de(p);
        det*=1.+totdist*.02;
        if (d<det && shcol>0. && ref<.5)
        {
            d=det;
            vec3 n=normal(p);
            refcol=shade(p,dir,n);
            dir=reflect(dir,n);
            ref=1.;
        }
        if (d<det) break;
        totdist+=d;
        g+=exp(-.03*totdist)*gcol;
    }
    col=mix((.5+sm(12.,15.,iTime)*.5)*vec3(.8,.9,1.),shade(p,dir,normal(p)),exp(-.1*totdist))
        +pow(g*.012,1.5)*vec3(1.2,.4,.2)+hcol*vec3(.7,.5,1.)*step(61.,iTime);
    return mix(col,refcol,ref*.85)+tcol*vec3(3.,1.,.5)+(h-.5)*.1;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin

    t=iTime*1.5-15.+sm(40.,47.,iTime)*10.-sm(107.,120.,iTime)*7.+sm(85.,90.,iTime)*2.;
    turbo=sm(108.,112.,iTime)*8.;
    vec2 uv = (fragCoord-iResolution.xy*.5)/iResolution.y;
    vec3 from = ro; //pathcam(t+.7*sign(iTime-40.5)*sign(58.-iTime)*sign(101.5-iTime)*(1.25-.5*sin(iTime/2.))), 
    vec3 to = ro; //pathcam(t+.2+turbo), 
    vec3 dir = rd; //normalize(vec3(uv,.7-sm(5.,0.,iTime)*.2));
    if (iTime<13.5) from=pathcam(9.5-iTime*.5),uv.y+=.0,from.y-=sm(10.,0.,iTime)*.8;
    if (abs(iTime-57.5) < 3.5) from = pathcam(82.+sin(iTime*.5));
    if (abs(iTime-71.) < 3.5 || abs(iTime-91.25) < 3.25) from.x += 5., from.z -=1.;
    shfrom = pathcam(t+turbo);
    shdir = normalize(to-shfrom);
    vec3 adv = normalize(to-from);
    adv.y*=sign(to.z-from.z);
    from.x+=shdir.x*.3*step(13.5,iTime);
    dir.yz=mul(dir.yz,rot(max(0.,.5-iTime*.07)));
    dir=mul(dir,lookat(adv,vec3(0.,1.,0.)));
    fragColor = vec4(march(from, dir)*min(1.,iTime*.5),1.0)*(1.-turbo*.15);

                return fragColor;
            }

            ENDCG
        }
    }
}


Shader "Skybox/PointlessStructures"
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


float det=.0001;
float maxdist=15.;
vec3 ldir=vec3(0.5,1.,1.);
vec3 pa;
float gcol;
float t, it, k;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}


float hash12(vec2 p)
{
    vec3 p3  = fract(vec3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

mat2 rot(float a) {
    a=radians(a);
    float s=sin(a),c=cos(a);
    return mat2(c,s,-s,c);
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

float kset(vec3 p) {
    p=abs(fract(p*.5)-.5);
    for (int i=0; i<6; i++) {
        p=abs(p)/dot(p,p)-.8;
    }
    return length(p.xy);
}


float shape (vec3 p, float z) {
    p.xz=mul(p.xz,rot(smoothstep(.28,.3,abs(.5-fract(t*.1+floor(z*4.)*.005)))*90.));
    float d=sdBoundingBox(p,vec3(1.,1.,2.),.07);
    return d;
}

vec3 path(float t) {
    return vec3(sin(t*.5),cos(t)*.5,t);
}

vec3 pathcam(float t) {
    vec3 p=path(t);
    p.y+=smoothstep(0.,.5,abs(.5-fract(t*.05)))*3.;
    return p;
}



float de(vec3 pos) {
    float tu=length(pos.xy-pathcam(pos.z).xy)-.1;
    pos.y+=-1.;
    pos.x-=.4;
    pos.xy-=path(pos.z).xy;
    float z=pos.z;
    pos=abs(4.-mod(pos,8.))-4.;
    pa=pos;
    float sc=1.4, d=1000., der=1.;
    vec3 p=pos,m=vec3(100.,100,100);
    float o=1000.;
    for (int i=0; i<7; i++) {
        p.xy=mul(p.xy,rot(90.));
        p.xz=abs(p.xz);
        p.y+=1.;
        sc=1.7/clamp(dot(p,p),0.1,1.0);
        p=p*sc-vec3(2.,1.,3.);
        p.y-=1.;
        der*=sc;
        float shp=shape(p,z)/der;
        if (shp<d && i>1) {
            d=shp;
            it=float(i);
        }
        o=min(o,length(p));
    }
    d=min(d,length(p.xy)/der-.005);
    gcol=step(fract(pos.z*.1+iTime*.2+p.z*.005+it*.25),.02)*10.+1.5;
    d=max(d,-tu);
    return d*.7;
}

vec3 march(vec3 from, vec3 dir) {
    vec3 p, col=vec3(0.,0,0);
    float totdist=0.,d;
    float g=0.,gg=0.;
    for(int i=0; i<130; i++) {
        p=from+totdist*dir;
        d=de(p);
        det*=1.+totdist*.03;
        if (d<det || totdist>maxdist) break;
        totdist+=d*(1.+hash12(dir.xy*1000.)*.3);
        g+=exp(-.03*totdist)*kset(p)*gcol;
    }
    if (d<.1) {
        col=gcol*vec3(.05,.05,.05);
    }
    col=mix(vec3(0.,0,0),col,exp(-.2*totdist));
    col=pow(col,vec3(1.3,1.3,1.3))*1.5;
    return col+pow(g*.012,1.5)*vec3(1.,.25,0.);
}

mat3 lookat(vec3 dir, vec3 up) {
    dir=normalize(dir); vec3 r=normalize(cross(dir,up));
    return mat3(r,cross(dir,r),dir);
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;
                float2 screenUV = v.screenPos.xy / v.screenPos.w;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+ _XYZPos;                                             // ray origin


    t=iTime*.5;
    vec2 uv = (fragCoord-iResolution.xy*.5)/iResolution.y;
    vec3 from = pathcam(t);
    vec3 to = pathcam(t+2.);
    vec3 adv = normalize(to-from);
    vec3 dir = normalize(vec3(uv,1.));
    dir=mul(dir,lookat(adv,vec3(0.,1.,0.)));
    dir.xy=mul(dir.xy,rot(45.));
    dir.yz=mul(dir.yz,rot(-20.));
    vec3 col = march(from, dir)*max(mod(fragCoord.x,3.),mod(fragCoord.y,3.))*.7;
    fragColor = vec4(col,1.0);

                return fragColor;
            }

            ENDCG
        }
    }
}



Shader "Skybox/JetStream"
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

// srtuss, 2015
//
// Volumetric cloud tunnel, a single light source, lightning and raindrops.
//
// The code is a bit messy, but in this case it's visuals that count. :P


#define pi 3.1415926535897932384626433832795

struct ITSC
{
    float3 p;
    float dist;
    float3 n;
    float2 uv;
};

ITSC raycylh(float3 ro, float3 rd, float3 c, float r)
{
    ITSC i;
    i.dist = 1e38;
    float3 e = ro - c;
    float a = dot(rd.xy, rd.xy);
    float b = 2.0 * dot(e.xy, rd.xy);
    float cc = dot(e.xy, e.xy) - r;
    float f = b * b - 4.0 * a * cc;
    if(f > 0.0)
    {
        f = sqrt(f);
        float t = (-b + f) / (2.0 * a);
        
        if(t > 0.001)
        {
            i.dist = t;
            i.p = e + rd * t;
            i.n = -float3(normalize(i.p.xy), 0.0);
        }
    }
    return i;
}

void tPlane(inout ITSC hit, float3 ro, float3 rd, float3 o, float3 n, float3 tg, float2 si)
{
    float2 uv;
    ro -= o;
    float t = -dot(ro, n) / dot(rd, n);
    if(t < 0.0)
        return;
    float3 its = ro + rd * t;
    uv.x = dot(its, tg);
    uv.y = dot(its, cross(tg, n));
    if(abs(uv.x) > si.x || abs(uv.y) > si.y)
        return;
    
    //if(t < hit.dist)
    {
        hit.dist = t;
        hit.uv = uv;
    }
    return;
}


float hsh(float x)
{
    return frac(sin(x * 297.9712) * 90872.2961);
}

float nseI(float x)
{
    float fl = floor(x);
    return lerp(hsh(fl), hsh(fl + 1.0), smoothstep(0.0, 1.0, frac(x)));
}

float2 rotate(float2 p, float a)
{
    return float2(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a));
}

float nse3d(in float3 x)
{
    float3 p = floor(x);
    float3 f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    float2 uv = (p.xy + float2(37.0, 17.0) * p.z) + f.xy;
    float2 rg = tex2D( _MainTex, (uv+.5)/256.).yx;               //tex2DLod( _MainTex, (uv+.5)/256., 0.).yx;
    return lerp(rg.x, rg.y, f.z);
}

float nse(float2 p)
{
    return tex2D(_MainTex, p).x;
}

float density2(float2 p, float z, float t)
{
    float v = 0.0;
    float fq = 1.0, am = 0.5, mvfd = 1.0;
    float2 rnd = float2(0.3, 0.7);
    for(int i = 0; i < 7; i++)
    {
        rnd = frac(sin(rnd * 14.4982) * 2987253.28612);
        v += nse(p * fq + t * (rnd - 0.5)) * am;
        fq *= 2.0;
        am *= 0.5;
        mvfd *= 1.3;
    }
    return v * exp(z * z * -2.0);
}

float densA = 1.0, densB = 2.0;

float fbm(float3 p)
{
    float3 q = p;
    //q.xy = rotate(p.xy, _Time.y);
    
    p += (nse3d(p * 3.0) - 0.5) * 0.3;
    
    //float v = nse3d(p) * 0.5 + nse3d(p * 2.0) * 0.25 + nse3d(p * 4.0) * 0.125 + nse3d(p * 8.0) * 0.0625;
    
    //p.y += _Time.y * 0.2;
    
    float mtn = _Time.y * 0.15;
    
    float v = 0.0;
    float fq = 1.0, am = 0.5;
    for(int i = 0; i < 6; i++)
    {
        v += nse3d(p * fq + mtn * fq) * am;
        fq *= 2.0;
        am *= 0.5;
    }
    return v;
}

float fbmHQ(float3 p)
{
    float3 q = p;
    q.xy = rotate(p.xy, _Time.y);
    
    p += (nse3d(p * 3.0) - 0.5) * 0.4;
    
    //float v = nse3d(p) * 0.5 + nse3d(p * 2.0) * 0.25 + nse3d(p * 4.0) * 0.125 + nse3d(p * 8.0) * 0.0625;
    
    //p.y += _Time.y * 0.2;
    
    float mtn = _Time.y * 0.2;
    
    float v = 0.0;
    float fq = 1.0, am = 0.5;
    for(int i = 0; i < 9; i++)
    {
        v += nse3d(p * fq + mtn * fq) * am;
        fq *= 2.0;
        am *= 0.5;
    }
    return v;
}

float density(float3 p)
{
    float2 pol = float2(atan2(p.y, p.x), length(p.yx));
    
    float v = fbm(p);
    
    float fo = (pol.y - 1.5);//(densA + densB) * 0.5);
    //fo *= (densB - densA);
    v *= exp(fo * fo * -5.0);
    
    float edg = 0.3;
    return smoothstep(edg, edg + 0.1, v);
}

float densityHQ(float3 p)
{
    float2 pol = float2(atan2(p.y, p.x), length(p.yx));
    
    float v = fbmHQ(p);
    
    float fo = (pol.y - 1.5);//(densA + densB) * 0.5);
    //fo *= (densB - densA);
    v *= exp(fo * fo * -5.0);
    
    float edg = 0.3;
    return smoothstep(edg, edg + 0.1, v);
}

float2 drop(inout float2 p)
{
    float2 mv = _Time.y * float2(0.5, -1.0) * 0.15;
    
    float drh = 0.0;
    float hl = 0.0;
    
    float4 rnd = float4(0.1, 0.2, 0.3, 0.4);
    for(int i = 0; i < 20; i++)
    {
        rnd = frac(sin(rnd * 2.184972) * 190723.58961);
        float fd = frac(_Time.y * 0.2 + rnd.w);
        fd = exp(fd * -4.0);
        float r = 0.025 * (rnd.w * 1.5 + 1.0);
        float sz = 0.35;
        
        
        float2 q = (frac((p - mv) * sz + rnd.xy) - 0.5) / sz;
        mv *= 1.06;
        
        q.y *= -1.0;
        float l = length(q + pow(abs(dot(q, float2(1.0, 0.4))), 0.7) * (fd * 0.2 + 0.1));
        if(l < r)
        {
            float h = sqrt(r * r - l * l);
            drh = max(drh, h * fd);
        }
        hl += exp(length(q - float2(-0.02, 0.01)) * -30.0) * 0.4 * fd;
    }
    p += drh * 5.0;
    return float2(drh, hl);
}


float hash1(float p)
{
    return frac(sin(p * 172.435) * 29572.683) - 0.5;
}

float hash2(float2 p)
{
    float2 r = (456.789 * sin(789.123 * p.xy));
    return frac(r.x * r.y * (1.0 + p.x));
}

float ns(float p)
{
    float fr = frac(p);
    float fl = floor(p);
    return lerp(hash1(fl), hash1(fl + 1.0), fr);
}

float fbm(float p)
{
    return (ns(p) * 0.4 + ns(p * 2.0 - 10.0) * 0.125 + ns(p * 8.0 + 10.0) * 0.025);
}

float fbmd(float p)
{
    float h = 0.01;
    return atan2(fbm(p + h) - fbm(p - h), h);
}

float arcsmp(float x, float seed)
{
    return fbm(x * 3.0 + seed * 1111.111) * (1.0 - exp(-x * 5.0));
}

float arc(float2 p, float seed, float len)
{
    p *= len;
    //p = rotate(p, _Time.y);
    float v = abs(p.y - arcsmp(p.x, seed));
    v += exp((2.0 - p.x) * -4.0);
    v = exp(v * -60.0) + exp(v * -10.0) * 0.6;
    //v += exp(p.x * -2.0);
    v *= smoothstep(0.0, 0.05, p.x);
    return v;
}

float arcc(float2 p, float sd)
{
    float v = 0.0;
    float rnd = frac(sd);
    float sp = 0.0;
    v += arc(p, sd, 1.0);
    for(int i = 0; i < 4; i ++)
    {
        sp = rnd + 0.01;
        float2 mrk = float2(sp, arcsmp(sp, sd));
        v += arc(rotate(p - mrk, fbmd(sp)), mrk.x, mrk.x * 0.4 + 1.5);
        rnd = frac(sin(rnd * 195.2837) * 1720.938);
    }
    return v;
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin

    float2 uv = fragCoord.xy; // / iResolution.xy;
    
    uv = 2.0 * uv - 1.0;
    //uv.x *= iResolution.x / iResolution.y;
    
    float2 drh = drop(uv);
    
    float camtm = _Time.y * 0.15;
     ro = float3(cos(camtm), 0.0, camtm);
    //float3 rd = normalize(float3(uv, 1.2));
    rd.xz = rotate(rd.xz, sin(camtm) * 0.4);
    rd.yz = rotate(rd.yz, sin(camtm * 1.3) * 0.4);
    
    float3 sun = normalize(_SunDir); //float3(0.2, 1.0, 0.1));
    
    float sd = sin(fragCoord.x * 0.01 + fragCoord.y * 3.333333333 + _Time.y) * 1298729.146861;
    
    float3 col;
    float dacc = 0.0, lacc = 0.0;
    
    float3 light = float3(cos(_Time.y * 8.0) * 0.5, sin(_Time.y * 4.0) * 0.5, ro.z + 4.0 + sin(_Time.y));
    
    ITSC tunRef;
    #define STP 15
    for(int i = 0; i < STP; i++)
    {
        ITSC itsc = raycylh(ro, rd, float3(0.0,0,0), densB + float(i) * (densA - densB) / float(STP) + frac(sd) * 0.07);
        float d = density(itsc.p);
        float3 tol = light - itsc.p;
        float dtol = length(tol);
        tol = tol * 0.1 / dtol;
        
        float dl = density(itsc.p + tol);
        lacc += max(d - dl, 0.0) * exp(dtol * -0.2);
        dacc += d;
        tunRef = itsc;
    }
    dacc /= float(STP);
    ITSC itsc = raycylh(ro, rd, float3(0.0,0,0), 4.0);
    float3 sky = float3(0.6, 0.3, 0.2);
    sky *= 0.9 * pow(fbmHQ(itsc.p), 2.0);
    lacc = max(lacc * 0.3 + 0.3, 0.0);
    float3 cloud = pow(float3(lacc,lacc,lacc), float3(0.7, 1.0, 1.0) * 1.0);
    col = lerp(sky, cloud, dacc);
//    col *= exp(tunRef.dist * -0.1);
//    col += drh.y;
    
    float4 rnd = float4(0.1, 0.2, 0.3, 0.4);
    float arcv = 0.0, arclight = 0.0;
    for(int i = 0; i < 3; i++)
    {
        float v = 0.0;
        rnd = frac(sin(rnd * 1.111111) * 298729.258972);
        float ts = rnd.z * 4.0 * 1.61803398875 + 1.0;
        float arcfl = floor(_Time.y / ts + rnd.y) * ts;
        float arcfr = frac(_Time.y / ts + rnd.y) * ts;
        
        ITSC arcits;
        arcits.p = float3(0,0,0);
        arcits.n =float3(0,0,0);
        arcits.uv = float2(0,0);
        arcits.dist = 1e38;
        float arca = rnd.x + arcfl * 2.39996;
        float arcz = ro.z + 1.0 + rnd.x * 12.0;
        tPlane(arcits, ro, rd, float3(0.0, 0.0, arcz), float3(0.0, 0.0, -1.0), float3(cos(arca), sin(arca), 0.0), float2(2.0,2.0));

        float arcseed = floor(_Time.y * 17.0 + rnd.y);
        if(arcits.dist < 20.0)
        {
            arcits.uv *= 0.8;
            v = arcc(float2(1.0 - abs(arcits.uv.x), arcits.uv.y * sign(arcits.uv.x)) * 1.4, arcseed * 0.033333);
        }
        float arcdur = rnd.x * 0.2 + 0.05;
        float arcint = smoothstep(0.1 + arcdur, arcdur, arcfr);
        v *= arcint;
        arcv += v;
        arclight += exp(abs(arcz - tunRef.p.z) * -0.3) * frac(sin(arcseed) * 198721.6231) * arcint;
    }
    float3 arccol = float3(0.9, 0.7, 0.7);
//    col += arclight * arccol * 0.5;
//    col = lerp(col, arccol, clamp(arcv, 0.0, 1.0));
//    col = pow(col, float3(1.0, 0.8, 0.5) * 1.5) * 1.5;
   // col = pow(col, float3(1.0 / 2.2,1.0 / 2.2,1.0 / 2.2));
    fragColor = float4(col, 1.0);



                return fragColor;
            }

            ENDCG
        }
    }
}




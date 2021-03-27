
Shader "Skybox/PlanetMars"

{
    Properties
    {
        _MainTex0 ("Texture", 2D) = "white" {}
        _MainTex1 ("Texture", 2D) = "white" {}
        _SkyColor ("Sky Color", Color) = (0.6, 0.25, 0.0, 0.0) 
        _CloudColor ("Cloud Color", Color) = (1.1, 0.6, 0.4, 1.0) 
        _RayColor ("Ray Color", Color) = (1.0, 1.0, .92, .0) 
        _SunColor ("Ray Color", Color) = (1.0, 1.0, .92, .0) 
        _GroundColor ("Ground Color", Color) = (.95, 0.7, 0.7, 0.0) 
        _FogColor ("Fog Color", Color) = (.2, .09, .0, .0) 
        _CloudFilter ("Cloud Filter", Float ) = 0.24
        _RayFilter ("Ray Filter", Float ) = 0.78
        _RayStrength ("Ray Strength", Float ) = 2.8
        _SunAngle ("Sun Angle", Float ) = 0.5
        _SunPos ("Sun Pos", Color) = (0.0, 5.0, 21.0) 
 
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

            uniform sampler2D _MainTex0; 
            uniform sampler2D _MainTex1; 
//            float4 _SkyColor;
//            float4 _CloudColor;
//            float4 _RayColor;
            float4 _SunColor;
            float4 _GroundColor;
            float4 _SunPos;
            float4 _FogColor;
            float _CloudFilter,_RayFilter,_RayStrength,_SunAngle;


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

float3x3 rotx(float a) { float3x3 rot; rot[0] = float3(1.0, 0.0, 0.0); rot[1] = float3(0.0, cos(a), -sin(a)); rot[2] = float3(0.0, sin(a), cos(a)); return rot; }
float3x3 roty(float a) { float3x3 rot; rot[0] = float3(cos(a), 0.0, sin(a)); rot[1] = float3(0.0, 1.0, 0.0); rot[2] = float3(-sin(a), 0.0, cos(a)); return rot; }
float3x3 rotz(float a) { float3x3 rot; rot[0] = float3(cos(a), -sin(a), 0.0); rot[1] = float3(sin(a), cos(a), 0.0); rot[2] = float3(0.0, 0.0, 1.0); return rot; }

float flter(float f, float a)
{
    f = clamp(f - a, 0.0, 1.0);
    return f / (1.0 - a);
}


float fbm(float2 uv)
{
    float f1 = (tex2D(_MainTex0, uv).r - 0.5)  * 0.5;
    f1 += (tex2D(_MainTex0, uv * 2.0).r - 0.5) * 0.25;
    f1 += (tex2D(_MainTex0, uv * 4.0).r - 0.5) * 0.25 * 0.5;
    f1 += (tex2D(_MainTex0, uv * 8.0).r - 0.5) * 0.25 * 0.5 * 0.5;
    f1 += (tex2D(_MainTex0, uv * 16.0).r - 0.5) * 0.25 * 0.5 * 0.5 * 0.5;
    return clamp(f1 + 0.5, 0.0, 1.0);
}


float clouds(in float3 rp, in float3 rd, float f)
{
    float t = _Time.y + 14.0;
    rd.y += rp.z;
    rd *= 0.15;
    rd.y += t * 0.1;
    
    float f1 = flter(fbm(rd.xz * 0.1), f) * .65;
    rd.x += t * 0.05;
    rd.y += t * 0.05;
    
    float f2 = flter(fbm(rd.xz * 0.45), f) * 0.25;
    rd.x += t * 0.01;
    rd.y += t * 0.01;

    float f3 = flter(fbm(rd.xz * 0.85), f) * .1;
    
    return clamp(f1 + f2 + f3, 0.0, 1.0);
}

float4 _SkyColor = float4(0.6, 0.25, 0.0, 0.0);
float4 _CloudColor = float4(1.1, 0.6, 0.4, 1.0);
float4 _RayColor = float4(1.0, 1.0, .92, .0);
//float4 groundcol = float4(.95, 0.7, 0.7, 0.0);

// godray and cloud controls
//const float cloudFilter = 0.24;
//const float rayFilter = 0.78;
//const float rayStrength = 2.0;

// 
const int steps = 40;
const int cloudsteps = 10;


void ground(in float3 rp, in float3 rd, inout float4 color)
{
    float3 ro = rp;
    float ydif = 0.25;
    rp += rd * (ydif / abs(rd.y));
    float scale = 0.3;
    
    for (int i = 0; i < 10; ++i)
    {
        float4 col = tex2D(_MainTex1, rp.xz * scale);
        float h = col.r * 0.3;
        rp += rd * h * 0.05;
        if(rp.y - h < -0.75) break;
    }
    color += tex2D(_MainTex1, rp.xz * scale) * _GroundColor;
}
const float mouseSpeedX = 2.0;



         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                //fixed4 col = tex2D(_MainTex0, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 rp = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

//    float2 uv = v.uv; //fragCoord.xy / iResolution.xy;
  //  uv = rd.xy;
 //       uv -= float2(0.5,0.5);

    float2  m = float2(0.0, -0.5);
    float3 _rd = rd;
    
    // 
    float4 col = float4(0.0,0,0,0);
    float4 finalcolor = float4(0.1, 1.0, 1.0, 0.0);
    
    float stepsf = float(steps);
    float cl = clouds(rp, rd / abs(rd.y), _CloudFilter);
    finalcolor = mul(cl , _CloudColor);
    finalcolor = lerp(finalcolor, _SkyColor, 1.0 - cl);
    
    // sun
    float3 sunpos = (_SunPos.xyz*2.0)-float3(1,1,1); //float3(0.0, 5.0 + sin(_Time.y * 0.25) * 3.0, 21.0);
    float3 sd = normalize(sunpos); //mul(normalize(sunpos) , rotx(-_SunAngle));
    
//    sd = mul(sd,roty(m.x * -mouseSpeedX));
    float2 sp = float2(sd.x / sd.z, sd.y / sd.z);
    float4 suncol = (1.0 - smoothstep(0.05, 0.1, length(rd-sd))) * _RayColor;
    suncol += 0.5 * (1.0 - smoothstep(0.02, 0.3, length(rd - sd)) * _RayColor);
    suncol.a = 0.0;
    
    finalcolor += suncol;

    // clouds
    float4 cloudcl = finalcolor;
    for (int j = 0; j < cloudsteps; ++j)
    {   
        float fi = float(j + 1);
        float fl = _CloudFilter + fi * 0.01;
        float fin = fi / stepsf;
        float c = clouds(rp, rd / abs(rd.y) + 0.2 * fi, fl);
        cloudcl = lerp(cloudcl, mul(_CloudColor , c), cloudcl.a);
    }
    finalcolor = cloudcl;
    // godrays        
    for (int k = 0; k < steps; ++k)
    {   
        float fi = float(k);
        float fin = fi / stepsf;

        float offset = 1.0 - fi * .01;
        float fl = _CloudFilter - fi * 0.0001;

        float rlen = offset / abs(rd.y);

        float3 projpos = rd * rlen;
        float3 dif = projpos - sunpos;

        dif = normalize(dif);
        dif = dif*-offset;
        rd = normalize(rd * rlen + dif * 0.035);
        rd /= abs(rd.y);

        float c = clouds(rp, rd,  fl);
        c = 1.0 - c;
        c = flter(c, _RayFilter);
        finalcolor += _RayColor * c  * (1.0 / stepsf) * _RayStrength * (1.0 - finalcolor.a) * (1.0 - fin);
    }
    col += finalcolor;
    
    // ground
    if(rd.y < 0.0)
    {
        float4 gc = float4(0.0,0,0,0);
        ground(rp, _rd, gc);
        col = lerp(gc, finalcolor, .3);
        col = mul(col,cloudcl.r);
    }
    
    float dst = length(float2(_rd.x / _rd.y, _rd.z / _rd.y));
    float4 fog = _SkyColor;
    fog = lerp(fog, float4(_FogColor), 1.0 - pow(abs(_rd.y), 0.7));
    col = lerp(col, fog, smoothstep(5.0, 25.0, dst) * 0.7);
    
    // mountains
    if(rd.y > 0.0)
    {
        float x = _rd.x;
        float y = _rd.y;
        
        float s = smoothstep(0.0, 1.0, sin(x * 2.0) * 0.5 + 0.5) * 0.03;
        s += smoothstep(0.0, 1.0, sin(x * 10.0) * 0.5 + 0.5) * 0.02;
        s += smoothstep(0.0, 1.0, sin(x * 20.0) * 0.5 + 0.5) * 0.01;
        s += smoothstep(0.0, 1.0, cos(x * 40.0) * 0.5 + 0.5) * 0.005;
        
        float l = smoothstep(0.0, 0.005, y - s);
        col = lerp(lerp(fog * 0.85, col, 0.1), col, l);;
    }
    float4 fragColor = col;
    //fragColor = mul(fragColor,float4(1.0, 1.0, 1.0, 1.0));

                return fragColor;
            }

            ENDCG
        }
    }
}





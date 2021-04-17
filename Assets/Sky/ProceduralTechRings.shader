
Shader "Skybox/ProceduralTechRings"
{
    Properties
    {
        _Sound ("_Sound", 2D) = "white" {}
        _Beat ("_Beat", float) = 0.0
         _Volume ("_Volume", float) = 0.0
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

            uniform sampler2D _Sound; 
            uniform int _SoundArrayLength = 256;
            float _Beat;
            uniform float _SoundArray[256];
            uniform float _Volume;

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

// srtuss, 2013

#define PI 3.14159265358979323

float2 rotate(float2 p, float a)
{
    return float2(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a));
}

float fft(float band)
{
//    return clamp(0,.8,_SoundArray[floor(band)]*1000/max(.01,_Volume));
    return clamp(0,.8,_SoundArray[floor(band)]*10000);
    //return tex2D( _Sound, float2(band,0.0) ).x; // Should sample sound stream not image
}

// iq's fast 3d noise tortured
float noise3(in float3 x)
{
    float3 p = floor(x);
    float3 f = frac(x);
    float3 nf = 3.0 - (2.0 * f);
    f = f * f * (nf);
    float2 uv = (p.xy + float2(37.0, 17.0+smoothstep(0.8,0.99, fft(0.))) * p.z) + f.xy;
//    float2 uv = (p.xy + float2(37.0, 17.0+smoothstep(0.8,0.99, fft(0.))) * p.z) + f.xy;

    float2 rg = tex2D(_Sound, (uv.x + 0.5) / 256.0).yx;  //, -100.0).yx;
    rg += tex2D(_Sound, (uv.x + _Time.y) / 64.0).yx/10.0;   //, -100.0).yx/10.0;
    rg += tex2D(_Sound, (uv.x + _Time.y/3.2 + 0.5) / 100.0).zx/5.0;    //, -100.0).zx/5.0;



    rg+=_SoundArray[floor(f.y*16)]*1;
    rg+=_SoundArray[floor(f.x*16)]*1;
    return lerp(rg.x, rg.y, f.z);
}

// 3d fbm
float fbm3(float3 p)
{
    return noise3(p) * 0.5 + noise3(p * 2.02) * 0.25 + noise3(p * 4.01) * 0.125;
}

// animated 3d fbm
float fbm3a(float3 p)
{
    float2 t = float2(_Time.y * 0.4, 0.0);
    return noise3(p + t.xyy) * 0.5 + noise3(p * 2.02 - t.xyy) * 0.25 + noise3(p * 4.01 + t.yxy) * 0.125;
}

// more animated 3d fbm
float fbm3a_(float3 p)
{
    float2 t = float2(_Time.y * 0.4, 0.0);
    return noise3(p + t.xyy) * 0.5 + noise3(p * 2.02 - t.xyy) * 0.25 + noise3(p * 4.01 + t.yxy) * 0.125 + noise3(p * 8.03 + t.yxy) * 0.0625;
}

// background
float3 sky(float3 p)
{
    float3 col;
    float v = 1.0 - abs(fbm3a(p * 4.0) * 2.0 - 1.0);
    float n = fbm3a_(p * 7.0 - 104.042);
    v = lerp(v, pow(n, 0.3), 0.8);
    
    col = float3(pow(float3(v,v,v), float3(14.0, 9.0, 7.0))) * 0.8;
    float ss = .00001+smoothstep(0.75,0.99,fbm3a_(p));
    col += float3(ss*8., 0.0, 0.0)*   fft(    floor(  (p.z+.5)  *10)     )*1000;
    return col;
}

         fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_Sound, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

//   float2 uv = fragCoord.xy / iResolution.xy;
//    uv = uv * 2.0 - 1.0;
//    uv.x *= iResolution.x / iResolution.y;
    
    float t = _Time.y;
    
    float3 dir = rd; //normalize(float3(uv, 1.1));
    
    dir.yz = rotate(dir.yz, sin(t/15.+smoothstep(0.99,0.999, fft(0.))));
    dir.xy = rotate(dir.xy, cos(t/13.));
//    dir.xz = rotate(dir.xz, cos(t/12.+smoothstep(0.5,0.999, fft(213.))));
    
    float3 col = sky(dir);

    // dramatize colors
    col = pow(col, float3(1.5,1.5,1.5)) * 2.0;

    fragColor = float4(col, 1.0);


                return fragColor;
            }

            ENDCG
        }
    }
}





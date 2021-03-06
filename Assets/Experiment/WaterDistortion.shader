Shader "Unlit/WaterDistortion" {
    Properties {
        _MainTex ("Main texture", 2D) = "white" {}
        _NoiseTex ("Noise texture", 2D) = "grey" {}
        _FGColor ("Foreground Color ", Color) = (.55, .75, 1., .5)
        _BGColor ("Background Color ", Color) = (.0, .3, .5, 1.)

        _Mitigation ("Distortion mitigation", Range(1, 30)) = 1
        _SpeedX("Speed along X", Range(0, 5)) = 1
        _SpeedY("Speed along Y", Range(0, 5)) = 1
    }

    SubShader {
       Tags { "Queue" = "Transparent" "RenderType"="Transparent" }
       LOD 100
           Blend SrcAlpha OneMinusSrcAlpha
 
        Pass {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            sampler2D _NoiseTex;
            float _SpeedX;
            float _SpeedY;
            float _Mitigation;

            struct v2f {
                half4 pos : SV_POSITION;
                half2 uv : TEXCOORD0;
            };

            fixed4 _MainTex_ST;
            float4 _FGColor, _BGColor;


            v2f vert(appdata_base v) {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);
                return o;
            }

            half4 frag(v2f i) : COLOR {
                half2 uv = i.uv;
                half noiseVal = tex2D(_NoiseTex, uv).r;
                uv.x = uv.x + noiseVal * sin(_Time.y * _SpeedX) / _Mitigation;
                uv.y = uv.y + noiseVal * sin(_Time.y * _SpeedY) / _Mitigation;
                float4 color = tex2D(_MainTex, uv) * _FGColor;
                return color;
            }

            ENDCG
        }
    }
    FallBack "Diffuse"
}
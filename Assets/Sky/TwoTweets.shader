
Shader "Skybox/TwoTweets"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            float f(float3 p) 
            { 
                p.z+=_Time.y;
                return length(.05*cos(9.*p.y*p.x)+cos(p)-.1*cos(9.*(p.z+.3*p.x-p.y)))-1.; 
            }

            fixed4 frag (v2f v) : SV_Target
            {
                float2 p = v.vertex;
                float4 c = float4(0.,0.,0.,1.);

                float3 d=.5-float3(p,1)/800.0,o=d;for(int i=0;i<128;i++)o+=f(o)*d;
                c.xyz = abs(f(o-d)*float3(0,1,2)+f(o-.6)*float3(2,1,0))*(1.-.1*o.z);
                
                return c;
            }





            ENDCG
        }
    }
}

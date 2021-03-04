Shader "Skybox/minimal"
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
                // float3 worldPosition = mul(unity_ObjectToWorld, float4(v.vertex.xyz, 1.0)).xyz;
                // o.cameraRelativeWorldPosition = worldPosition - _WorldSpaceCameraPos.xyz;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // sample the texture
                fixed4 col = tex2D(_MainTex, i.uv);
                // fixed4 col = texCUBE(_CubeMap, i.cameraRelativeWorldPosition)

                if (i.uv.x > 0.0 && i.uv.x < 0.01  || i.uv.y > 0.0 && i.uv.y < 0.01) col.rb = 0.;
                float xx = (i.uv.x) % 0.02;
                if ( xx > -0.001 && xx < 0.001) col.g = 0.;
                //if (i.uv.z > 0.0 && i.uv.z < 0.01) col.rb = 0.;
                
                // apply fog
//                UNITY_APPLY_FOG(i.fogCoord, col);
                return col;
            }
            ENDCG
        }
    }
}

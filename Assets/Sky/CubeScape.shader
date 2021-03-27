
Shader "Skybox/CubeScape"
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

            fixed4 frag (v2f v) : SV_Target
            {
                float2 U = v.vertex;
                float4 O = float4(0.,0.,0.,0.);
 
                float l = .2;
                float3 R = float3(800.,450.,0.),
                     P = float3(_Time.y,3.,0.)*7.,
                     D = float3( ( U - .5*R.xy ) / R.y + l*cos(.5*_Time.y) , 1 );
                  // D = float3( ( U - .5*R.xy ) / R.y                   , 1 ); // 1 tweet version
                for( D.yz = mul(D.yz,float2x2(4.,-3.,3.,4.)*l); 
                     l > .1; 
                     l = min( P.y - 8.* frac( sin(dot(ceil(P).xzz,R)) * 4e5 ), .11 )
                   ) P += l*D;
                O += P.y/P.z * length(step(.1,frac(P))) -O;

                crash // Broken ... causes Unity to freeze up
                return O;
            }

crash







            ENDCG
        }
    }
}

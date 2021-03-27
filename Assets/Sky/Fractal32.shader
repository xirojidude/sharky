
Shader "Skybox/Fractal32"
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
                float2 C = v.vertex;
                float4 O = float4(0.,0.,0.,0.);
                float3 p,r=float3(800.,450.,0.);
                float3 d=normalize(float3((C-.5*r.xy)/r.y,1));  
                for(float i=0.,g=0.,e,s;
                    ++i<99.;
                    e<.001?O+=3.*(cos(float4(3,8,25,0)+log(s)*.5)+3.)/dot(p,p)/i:O
                )
                {
                    p=mul(g,d);
                    p-=float3(0,-.9,1.5);
                    r=normalize(float3(1,8,0));
                    s=_Time.y;//*.2;
                    p=lerp(r*dot(p,r),p,cos(s))+sin(s)*cross(p,r);
                    s=2.;
                    s=mul(s,e=3./min(dot(p,p),20.));
                    p=abs(p)*e;
                    for(int j=0;j++<4;)
                        p=float3(2,4,2)-abs(p-float3(4,4,2)),
                        s=mul(s,e=8./min(dot(p,p),9.))  ,
                        p=abs(p)*e;
                    g+=e=min(length(p.xz)-.15,p.y)/s;
                }
               
                return O;
            }

            ENDCG
        }
    }
}

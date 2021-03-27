
Shader "Skybox/MysteryMountains"
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

            uniform sampler2D _MainTex; 
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }
//            struct appdata {
//                float4 vertex : POSITION;
//            };
//            struct v2f {
//                float4 uv : TEXCOORD0;         //posWorld
//                float4 vertex : SV_POSITION;   //pos
//            };
//            v2f vert (appdata v) {
//                appdata v2;
//                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
//                v2f o = (v2f)0;
//                o.uv = mul(unity_ObjectToWorld, v2.vertex);
//                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
//                return o;
//            }

            // Add texture layers of differing frequencies and magnitudes...
            #define F +tex2D(_MainTex,.3+p.xz*s/3e3)/(s+=s) 

           fixed4 frag (v2f v) : SV_Target
            {
                float2 w = v.uv;
                float4 c = float4(0.,0.,0.,1.);
                float4 p=float4(w.x,w.y,1.,1.)-.5,d=p,t;
                p.z += _Time.y*20.;d.y-=.4;
                
                for(float i=1.5;i>0.;i-=.02)
                {
                    float s=.5;
                    t = F F F F F F;
                    c =1.+d.x-t*i; c.z-=.1;
                    if(t.x>p.y*.007+1.3)break;
                    p += d;
                }
                return c;
            }


//// [2TC 15] Mystery Mountains.
// David Hoskins.

            ENDCG
        }
    }
}

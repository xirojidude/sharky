
Shader "Skybox/TwoTweets"
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

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };


            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _SunDir,_XYZPos;

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


            float f(float3 p) 
            { 
                p.z+=_Time.y;
                return length(.05*cos(9.*p.y*p.x)+cos(p)-.1*cos(9.*(p.z+.3*p.x-p.y)))-1.; 
            }

            fixed4 frag (v2f v) : SV_Target
            {
                 float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin

                float2 p = v.vertex;
                float4 c = float4(ro,1); //float4(0.,0.,0.,1.);

                float3 d=.5-float3(p,1)/800.0,o=d;for(int i=0;i<128;i++)o+=f(o)*d;
                c.xyz = abs(f(o-d)*float3(0,1,2)+f(o-.6)*float3(2,1,0))*(1.-.1*o.z);
                
                return c;
            }





            ENDCG
        }
    }
}

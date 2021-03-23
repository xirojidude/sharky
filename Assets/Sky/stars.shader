Shader "Skybox/stars"
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
            #define UNITY_PASS_FORWARDBASE
            #include "UnityCG.cginc"
            #pragma multi_compile_fwdbase_fullshadows
            #pragma only_renderers d3d9 d3d11 glcore gles n3ds wiiu 
            #pragma target 3.0


            #include "UnityCG.cginc"

            sampler2D _MainTex;


        float time = 0.0;
             
        float3 rotate(float3 r, float v){ return float3(r.x*cos(v)+r.z*sin(v),r.y,r.z*cos(v)-r.x*sin(v));}

        float noise( in float3 x )
        {
            float  z = x.z*64.0;
            float2 offz = float2(0.317,0.123);
            float2 uv1 = x.xy + offz*floor(z); 
            float2 uv2 = uv1  + offz;
            return lerp(tex2D(_MainTex,  uv1 ).x,tex2D(_MainTex,  uv2 ).x,frac(z))-0.5;
        }

        float noises( in float3 p){
            float a = 0.0;
            for(float i=1.0;i<3.0;i++){
                a += noise(p)/i;
                p = p*2.0 + float3(0.0,a*0.001/i,a*0.0001/i);
            }
            return a;
        }

 

            

           struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float4 posWorld : TEXCOORD0;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                return o;
            }
 

            fixed4 frag (v2f i) : SV_Target
            {   
                float3 viewDirection = normalize(i.posWorld.xyz- _WorldSpaceCameraPos.xyz  );
                float3 worldPos = i.posWorld.xyz;//_WorldSpaceCameraPos.xyz;

                float3 finalColor = float4(0.5,0.4,0.6,0.);

                time        = _Time.y*5.0+floor(_Time.y*0.1)*150.0;
                time = 0.;
                float2 uv     = i.posWorld.xy; //fragCoord.xy/(iResolution.xx*0.5)-float2(1.0,iResolution.y/iResolution.x);
                float3 ray = viewDirection;
                
                float color = noises(ray);
                color = (color<.1)?0.:color;
                float4 fragColor = float4(color,color,color,1.);
                return fragColor;
                

            }

            ENDCG
        }
    }
}

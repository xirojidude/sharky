
Shader "Skybox/Fissure"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _GroundColor ("Ground Color", Color) = (.3,.7,1,0)
        _SkyColor ("Sky Color", Color) = (.3,.3,.3,1)
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

            uniform sampler2D _MainTex; 
            float4 _GroundColor,_SkyColor;

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


         fixed4 frag (v2f v) : SV_Target
            {
                float2 w = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 f = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    float4 p = float4(ro.xz,0,1/450);//,d=p,t; ////float4(w,0.,1.)/iResolution.xyxy-.5, 
    float4 d= float4(rd,1), t;
    float s = sign(d.y);
    p.z -= _Time.y;//*4.;
    d.y = -abs(d.y);
    for(float i = 1.5; i >0.; i-=.01)
    {
        t = max(s>0?_GroundColor:_SkyColor, tex2D(_MainTex, p.xz * .001 * s ));
        f = t*i;
        t.x -= p.y*.04;
        if(t.x>.99) break;
        p += (d-d*t.x)*8.;
    }

                return f;
            }

            ENDCG
        }
    }
}


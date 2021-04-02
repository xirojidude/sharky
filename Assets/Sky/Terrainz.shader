
Shader "Skybox/Terrainz"
{
    Properties
    {
        _MainTex0 ("tex2D", 2D) = "white" {}
        _MainTex1 ("tex2D", 2D) = "white" {}
        _XYZPos ("XYZ Offset", Vector) =  (0, 15, -.25 ,0) 
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

            uniform sampler2D _MainTex0; 
            uniform sampler2D _MainTex1; 
            float4 _XYZPos;

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

float3 L(float3 d) {
    return lerp(float3(0,.5,1), float3(1,1,1), pow(max(dot(d, float3(0, 1, 0)), 0.), .2));
}


         fixed4 frag (v2f v) : SV_Target
            {
                float2 f = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 c = tex2D(_MainTex0, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*-.01+_XYZPos;                                             // ray origin
//    float2 R=iResolution.xy,u = f/R;
    float2 u = float2(1,1);
    
    float3 d = rd; //float3(u*2.-1., 1);
   
    float3 po = ro; //float3(0, 4, iTime);
    float3 p = po;
    //d.y*=R.y/R.x;
     float3 m = L(d+float3(0, .6, 0));
    
    bool h = false;
    [loop]
    for (int i = 0; i < 100; ++i) {
        float f = p.y+tex2D(_MainTex0, p.xz/10.).r*2.+sin(p.x+p.z/10.)*cos(p.z-p.x/100.)-tex2D(_MainTex0, p.xz/100.).r*2.;
        if (length(p-po)>20.) break;
        if (f < .2) {
            h = true;
           
        }
        p+=d*f/5.;
    }
    if (h) float ln = .8-length(p-po)/20.;
    float3 c1 = tex2D(_MainTex1, p.xz).rgb;//+(float3(ln,ln,ln))/2.;
    float pw =pow(    length(p-po)/20., 25.);

    m = lerp(
        c1,
        m, pw)*(1.-tex2D(_MainTex0, p.xz/10.).r*.7);
 //       c = float4(m, 1);
    c = pow(float4(m, 1)*(1.-length(u-.5)/1.62), float4(1./1.5,1./1.5,1./1.5,1./1.5));

                return c;
            }

            ENDCG
        }
    }
}



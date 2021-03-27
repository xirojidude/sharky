
Shader "Skybox/Dynamism"
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

            uniform sampler2D _MainTex; 

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


const float2 center = float2(0,0);
const int samples = 15;
const float wCurveA = 1.;
const float wCurveB = 1.;
const float dspCurveA = 2.;
const float dspCurveB = 1.;

#define time iTime

float wcurve(float x, float a, float b)
{
    float r = pow(a + b,a + b)/(pow(a, a)*pow(b, b));
    return r*pow(x, a)*pow(1.0 - x, b);
}

float hash21(in float2 n){ return frac(sin(dot(n, float2(12.9898, 4.1414))) * 43758.5453); }

//void mainImage( out vec4 fragColor, in vec2 fragCoord )
        fixed4 frag (v2f v) : SV_Target
            {
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    float2 p = ro.xy; //fragCoord/iResolution.xy;
    float2 mo = rd.xy; //iMouse.xy/iResolution.xy;
    
    float2 center= mo;
    center = float2(0.5,0.5);
    
    float3  col = float3(0.0,0,0);
    float2 tc = center - p;
    
    float w = 1.0;
    float tw = 1.;
    
    float rnd = (hash21(p)-0.5)*0.75;
    
    //derivative of the "depth"
    //time*2.1 + ((1.0+sin(time + sin(time*0.4+ cos(time*0.1)))))*1.5
    float x = _Time.y;
    float drvT = 1.5 * cos(x + sin(0.4*x + cos(0.1*x)))*(cos(0.4*x + cos(0.1*x)) * (0.4 - 0.1*sin(0.1*x)) + 1.0) + 2.1;
    
    
    float strength = 0.01 + drvT*0.01;
    
    for(int i=0; i<samples; i++)
    {
        float sr = float(i)/float(samples);
        float sr2 = (float(i) + rnd)/float(samples);
        float weight = wcurve(sr2, wCurveA, wCurveB);
        float displ = wcurve(sr2, dspCurveA, dspCurveB);
        col += tex2D( _MainTex, p + (tc*sr2*strength*displ)).rgb*weight;
        tw += .9*weight;
    }
    col /= tw;

    fragColor = float4( col, 1.0 );

                return fragColor;
            }

            ENDCG
        }
    }
}

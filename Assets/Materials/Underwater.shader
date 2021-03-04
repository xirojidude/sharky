Shader "Unlit/Underwater"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _FGColor ("Foreground Color ", Color) = (.55, .75, 1., 1.)
        _BGColor ("Background Color ", Color) = (.0, .3, .5, 1.)
    }
    SubShader
    {
        Tags { "Queue" = "Transparent" "RenderType"="Transparent" }
        LOD 100
           Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            //#pragma surface alpha:fade
           //#pragma surface surf Standard fullforwardshadows alpha:fade

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _FGColor, _BGColor;


            #define tau 6.28318530718

            float sin01(float x) {
                return (sin(x*tau)+1.)/2.;
            }
            float cos01(float x) {
                return (cos(x*tau)+1.)/2.;
            }

           // rand func from theartofcode (youtube channel)
           float2 rand01(float2 p) {
           float3 a = frac(p.xyx * float3(123.5, 234.34, 345.65));
               a += dot(a, a);
        
               return frac (float2(a.x * a.y, a.y * a.z));
           }

           float circ(float2 uv, float2 pos, float r) {
               return smoothstep(r, 0., length(uv - pos));
           }

          float smoothFract(float x, float blurLevel) {
              return pow(cos01(x), 1./blurLevel);
           }

        float manDist(float2 from, float2 to) {
            return abs(from.x - to.x) + abs(from.y - to.y);
        }


        float distFn(float2 from, float2 to) {
            float x = length (from - to);
            return pow (x, 4.);
        }

        float voronoi(float2 uv, float t, float seed, float size) {
            
            float minDist = 100.;
            
            float gridSize = size;
            
            float2 cellUv = frac(uv * gridSize) - 0.5;
            float2 cellCoord = floor(uv * gridSize);
            
            for (float x = -1.; x <= 1.; ++ x) {
                for (float y = -1.; y <= 1.; ++ y) {
                    float2 cellOffset = float2(x,y);
                    
                    // Random 0-1 for each cell
                    float2 rand01Cell = rand01(cellOffset + cellCoord + seed);
                    
                    // Get position of point
                    float2 point1 = cellOffset + sin(rand01Cell * (t+10.)) * .5;
                    
                    // Get distance between pixel and point
                    float dist = distFn(cellUv, point1);
                    minDist = min(minDist, dist);
                }
            }
            
            return minDist;
        }

        fixed4 frag (v2f i) : SV_Target
        {
            // Center coordinates at 0
            float2 uv = float2(i.uv.x*2., i.uv.y);
            
            float t = _Time.y * .35;
            
            // Distort uv coordinates
            float amplitude = .12;
            float turbulence = .5;
            uv.xy += sin01(uv.x*turbulence + t) * amplitude;
            uv.xy -= sin01(uv.y*turbulence + t) * amplitude;
            
            // Apply two layers of voronoi, one smaller   
            float v=0.;
            float sizeDistortion = abs(uv.x)/5.;
            v += voronoi(uv, t * 2., 0.5, 2.5 - sizeDistortion);
            v += voronoi(uv, t * 4., 0., 4. - sizeDistortion) / 2.;
            
            // Foreground color
            float4 col = v * _FGColor; 
            
            // Background color
            col += (1.-v) * _BGColor;  
            
            // Output to screen
            return col;
        }
 
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            ENDCG
        }
    }
}

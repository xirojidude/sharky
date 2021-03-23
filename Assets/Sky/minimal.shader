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
            #define UNITY_PASS_FORWARDBASE
            #include "UnityCG.cginc"
            #pragma multi_compile_fwdbase_fullshadows
            #pragma only_renderers d3d9 d3d11 glcore gles n3ds wiiu 
            #pragma target 3.0


            uniform sampler2D _MainTex; 

            
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };


            struct VertexOutput {
                float4 pos : SV_POSITION;
                float4 posWorld : TEXCOORD0;
            };

            VertexOutput vert (appdata v) {
                VertexOutput o = (VertexOutput)0;
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
                o.pos = UnityObjectToClipPos( v.vertex); 
                return o;
            }
            



            //
            // Distance field function for the scene. It combines
            // the seperate distance field functions of three spheres
            // and a plane using the min-operator.
            //

            float map(float3 p) {
                float d = distance(p, float3(-10, 0, -50)) - 5.;     // sphere at (-1,0,5) with radius 1
                d = min(d, distance(p, float3(20, 0, -30)) - 10.);    // second sphere
                d = min(d, distance(p, float3(-20, 0, -20)) - 10.);   // and another
                d = min(d, p.y + 1.);                            // horizontal plane at y = -1
                return d;
            }

            //
            // Calculate the normal by taking the central differences on the distance field.
            //
            float3 calcNormal(in float3 p) {
                float2 e = float2(1.0, -1.0) * 0.0005;
                return normalize(
                    e.xyy * map(p + e.xyy) +
                    e.yyx * map(p + e.yyx) +
                    e.yxy * map(p + e.yxy) +
                    e.xxx * map(p + e.xxx));
            }

            

            float4 frag(VertexOutput i) : SV_TARGET //COLOR 
            {
                float3 viewDirection = normalize(i.posWorld.xyz- _WorldSpaceCameraPos.xyz  );

                float3 finalColor = float4(0.5,0.5,0.5,1.);

                float3 ro = _WorldSpaceCameraPos.xyz; //i.posWorld.xyz;                           // ray origin

                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy

                // March the distance field until a surface is hit.
                float h, t = 1.;
                for (int i = 0; i < 256; i++) {
                    h = map(ro + rd * t);
                    t += h;
                    if (h < 0.01) break;
                }

                if (h < 0.01) {
                    float3 p = ro + rd * t;
                    float3 normal = calcNormal(p);
                    float3 light = float3(0, 20, 0);
                    
                    // Calculate diffuse lighting by taking the dot product of 
                    // the light direction (light-p) and the normal.
                    float dif = clamp(dot(normal, normalize(light - p)), 0., 1.);
                    
                    // Multiply by light intensity (5) and divide by the square
                    // of the distance to the light.
                    dif *= 5. / dot(light - p, light - p);
                    
                    float shade = pow(dif, 0.4545);
                    float3 clr = float3(shade,shade,shade);
                    finalColor = float4(shade,shade,shade,1.);
                    
                } 

                return fixed4(finalColor,1);
            }
  
            ENDCG
        }
    }
}

Shader "Skybox/PanoSkybox" {
    Properties {
        _MainTex ("MainTex", 2D) = "white" {}
        [MaterialToggle] _StereoEnabled ("Stereo Enabled", Float ) = 0
    }
    SubShader {
        Tags {
            "RenderType"="Opaque"
        }
        Pass {
            Name "FORWARD"
            Tags {
                "LightMode"="ForwardBase"
            }
            
            
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #define UNITY_PASS_FORWARDBASE
            #include "UnityCG.cginc"
            #pragma multi_compile_fwdbase_fullshadows
            #pragma only_renderers d3d9 d3d11 glcore gles n3ds wiiu 
            #pragma target 3.0
            uniform sampler2D _MainTex; uniform float4 _MainTex_ST;

            float2 StereoPanoProjection( float3 coords ){
                float3 normalizedCoords = normalize(coords);
                float latitude = acos(normalizedCoords.y);
                float longitude = atan2(normalizedCoords.z, normalizedCoords.x);
                float2 sphereCoords = float2(longitude, latitude) * float2(0.5/UNITY_PI, 1.0/UNITY_PI);
                sphereCoords = float2(0.5,1.0) - sphereCoords;
                return (sphereCoords + float4(0, 1-unity_StereoEyeIndex,1,0.5).xy) * float4(0, 1-unity_StereoEyeIndex,1,0.5).zw;
            }
            
            float2 MonoPanoProjection( float3 coords ){
                float3 normalizedCoords = normalize(coords);
                float latitude = acos(normalizedCoords.y);
                float longitude = clamp(atan2(normalizedCoords.z, normalizedCoords.x),-3.1459,3.14159);
                float2 sphereCoords = float2(longitude, latitude) * float2(1.0/UNITY_PI, 1.0/UNITY_PI);
                sphereCoords = float2(1.0,1.) - sphereCoords;
                return (sphereCoords + float4(0, 1-unity_StereoEyeIndex,1,1.0).xy) * float4(0, 1-unity_StereoEyeIndex,1,1.0).zw;
            }
            
            uniform fixed _StereoEnabled;
            struct VertexInput {
                float4 vertex : POSITION;
            };
            struct VertexOutput {
                float4 pos : SV_POSITION;
                float4 posWorld : TEXCOORD0;
            };
            VertexOutput vert (VertexInput v) {
                VertexInput v2;
                v2.vertex = mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                VertexOutput o = (VertexOutput)0;
                o.posWorld = mul(unity_ObjectToWorld, v2.vertex);
                o.pos = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }
            
            #define TRANSFORM_TEX2(tex,name) (float2(0.5, 1.0) * tex.xy * name##_ST.xy + name##_ST.zw)

            float4 frag(VertexOutput i) : COLOR {
                float3 viewDirection = normalize(_WorldSpaceCameraPos.xyz - i.posWorld.xyz);
////// Lighting:
////// Emissive:
                float3 node_6020 = (viewDirection*(-1.0));
//                float2 _StereoEnabled_var = lerp( MonoPanoProjection( node_6020 ), StereoPanoProjection( node_6020 ), _StereoEnabled );
                float2 _StereoEnabled_var = MonoPanoProjection( node_6020 );
                float4 _MainTex_var = tex2D(_MainTex,TRANSFORM_TEX2(_StereoEnabled_var, _MainTex));
//                float4 _MainTex_var = tex2D(_MainTex,node_6020);
                float3 emissive = _MainTex_var.rgb;
                float3 finalColor = emissive;
                return fixed4(finalColor,1);
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
    CustomEditor "ShaderForgeMaterialInspector"
}

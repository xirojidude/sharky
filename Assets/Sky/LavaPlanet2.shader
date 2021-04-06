
Shader "Skybox/LavaPlanet"
{
    Properties
    {
        _MainTex ("tex2D", 2D) = "white" {}
        _MainTex1 ("tex2D", 2D) = "white" {}
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

// The red channel defines the height, but all channels
// are used for color.
#define elevationMap _MainTex
#define colorMap _MainTex1

           uniform sampler2D _MainTex; 
           uniform sampler2D _MainTex1; 
            float4 _SunDir,_XYZPos;

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
                float2 fragCoord = v.vertex;

                float3 viewDirection = normalize(v.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 outColor = tex2D(_MainTex, v.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz+_XYZPos;                                             // ray origin


    outColor = float4(0,0,0,0);

    // 'view' direction. This is the ray projected (effectively)
    // into tex2D space, with the y-axis keeping track of 
    float3 dir = rd; //float3(fragCoord / iResolution.x, 1) - 0.5;
    dir /= -abs(dir.y);
    
    // xz is the tex2D space offset. y is the distance from
    // the max height down to the surface.
    float3 offset = ro;//float3(0,0,0);
    float3 glow;
    float3 material;
    float3 landscape; 
    float2 scrollDirection = float2(0.707, 0.707);
        
    
    float2 coord = fragCoord;
    // Stack many thin slices. The 0.6 height is critical to
    // the character. The step size affects performance and quality,
    // but also brightness because there is no normalizing factor
    // on "glow". This loop is just for the volumetrics and finding the
    // surface. The surface color and shading is outside of the loop.
    //
    // Work from the top down (the surface is pushed in by parallax mapping
    // instead of popped up)
    float clearance = 1.0;
    for (float h = 0.0; h < 0.6 && clearance > 0.0; h += 0.01) {

        // Use the x channel of tex2D 0 as a threshold for elevation.
        // Once this falls to zero or below the rest of the loop
        // does nothing.
        clearance = max(0.0, tex2D(elevationMap, coord * 0.1).r - offset.y);
        if (clearance > 0.0) {
            // Volumetric lighting
            
            // Calculate offset of slice intersection
            // at this elevation.
            offset = ro+dir * h;
            
            // Add time sliding to move the camera
            coord = offset.xz; //- _Time.y * 0.2 * scrollDirection;
            
            landscape = tex2D(elevationMap, coord * 0.1).rgb;
            material = tex2D(colorMap, coord).rgb;
            
            // Don't glow too much near the top of mountains 
            // or we'll see banding in the lighting.
            glow = pow(landscape * material, material + 7.0 + 0.1) * offset.y;
            outColor.rgb += glow;
        }
    }
    
    
    // Term for over-lighting the surface to hide
    // banding artifacts. Increase contrast close to
    // the surface (to make lava glow), decrease higher
    // up (to hide banding).
    float contrast = offset.y;
    
    // Use "landscape" at this point as fake light + shadow
    // on the surface
    landscape -= tex2D(elevationMap, coord * 0.1 -  0.01).rgb - 0.4; 
    // Color the surface
    outColor.rgb += material * landscape + glow * 6e2 * contrast;
    
    // Darken in the distance
    outColor.rgb -= -offset.z * 0.03;

                return outColor;
            }

            ENDCG
        }
    }
}


// Deconstructed [SH17A] Lava Planet
// originally by @P_Malin for the "two tweet" challenge.
// https://www.shadertoy.com/view/ldBfDR
//
// I simplified some parts for clarity as well.
// This shader renders the planet by slices. It is very
// similar to parallax tex2D mapping, but it accumulates
// the volumetric glow while searching for the planet surface.
// The sky is exactly the same as the ground, but since you're
// looking up, it acts flat because the height threshold is never
// hit.



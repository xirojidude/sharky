//https://www.shadertoy.com/view/lsBfDz
Shader "Skybox/tinyCloud"
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

            struct VertexInput {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };
            struct VertexOutput {
                float4 vertex : SV_POSITION;
                float4 posWorld : TEXCOORD0;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            float2 MonoPanoProjection( float3 coords ){
                float3 normalizedCoords = normalize(coords);
                float latitude = 1.-(acos(normalizedCoords.y)/UNITY_PI);
                float longitude = 1.-clamp(atan2(normalizedCoords.z, normalizedCoords.x),-3.145926535,3.145926535)/UNITY_PI;

                float2 sphereCoords = float2(longitude, latitude);
                return (sphereCoords + float4(0, 1-unity_StereoEyeIndex,1,1.0).xy) * float4(0, 1-unity_StereoEyeIndex,1,1.0).zw;
            }

           #define TRANSFORM_TEX2(tex,name) (float2(0.5, 1.0) * tex.xy * name##_ST.xy + name##_ST.zw)

            VertexOutput vert (VertexInput v)
            {
//                VertexOutput o = (VertexOutput)0;
//                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
//                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
//                return o;
 
                VertexOutput o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                float2 uv = v.uv;  //TRANSFORM_TEX2(v.uv, _MainTex);
                o.posWorld = float4(uv.x,uv.y,1.,1.);
                return o;
            }

            #define T tex2D(_MainTex,(s*ray.zw+ceil(s*ray.x))/200.0).y/(s+=s)*4.

            fixed4 frag (VertexOutput i) : SV_Target
            {
                float3 viewDirection = normalize(i.posWorld.xyz- _WorldSpaceCameraPos.xyz  );
                float2 Pano = MonoPanoProjection( float3(viewDirection.x,viewDirection.y-1.,viewDirection.z) );

                float2 uv=i.posWorld;
                //*Note: GLSL texture coords are from -1 to +1  but we use 0 to +1 in HLSL 
                //float4 p,d=float4(.8,0,x/iResolution.y-.8),c=float4(.6,.7,d);

                //ray is ray position during ray march
                //ray.y is never used. 
                //ray.x is actually depth into the screen, 
                //ray.z is the screen x axis,  should go from -1 to +1
                //ray.w is the screen y axis (aka the up axis)   should go from -1 to +1
                float4 ray= float4(.8,0,uv.x-.8,uv.y-.8); 
                
                // d is direction that the ray for this pixel travels in 
                // 0.8 is subtracted from d.z and d.w (the screen x and screen y axes)
                // that makes the screen x axis 0 nearly centered on the screen. It also points the screen y axis downward a bit,
                // putting the 0 value near the top of the screen to make the camera look more downward at the clouds.
                float4 d= float4(.8,0,uv.x-.8,uv.y- (_WorldSpaceCameraPos.y*.01 -.01));

                // c is color of the sky
                // nice sky blue. It’s initialized with constants in x and y, and then d is used for z and w. d.xy goes into c.zw. 
                // That gives c the 0.8 value in the z field.
                //  Note that c.w is used to calculate O.w (O.a) but that the alpha channel of the output pixel value is currently ignored
                float4 c= float4(.6,.7,d.x,d.y);

                // O is the output color (initialized to blue sky)
                // subtracts d.w which is the pixel’s ray march direction on the screen y axis. 
                // This has a nice effect of making a nice sky blue gradient.
                float4 O=c-(d.w*.5);
                //O = ray;    

                for(float f,s,t=100.0+sin(dot(uv,uv))*5.;t>0.;t--) {
                    ray=.05*t*d;
                    ray.xz+=_Time.y*.05;
                    s=2.;
                    f=ray.w+1.-T-T-T-T;  // uses T macro from above
                    if(f<0.) {
                        O+=(O-1.-f*c.zyxw)*f*.2;
                    }
                }
                return O;
            }
           ENDCG
        }
    }
}

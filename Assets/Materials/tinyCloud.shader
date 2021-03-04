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

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

            #define T tex2D(_MainTex,(s*ray.zw+ceil(s*ray.x))/10.00).y/(s+=s)*4.

            fixed4 frag (v2f i) : SV_Target
            {
                float2 uv= float2(i.vertex);
//                float4 p,d=float4(.8,0,x/iResolution.y-.8),c=float4(.6,.7,d);

                //ray is ray position during ray march
                //ray.y is never used. 
                //ray.x is actually depth into the screen, 
                //ray.z is the screen x axis, 
                //ray.w is the screen y axis (aka the up axis)
                float4 ray= float4(.8,0,uv.x-.8,uv.y-.8);
                
                // d is direction that the ray for this pixel travels in 
                // 0.8 is subtracted from d.z and d.w (the screen x and screen y axes)
                // that makes the screen x axis 0 nearly centered on the screen. It also points the screen y axis downward a bit,
                // putting the 0 value near the top of the screen to make the camera look more downward at the clouds.
                float4 d= float4(.8,0,uv.x-.8,uv.y-.8);

                // c is color of the sky
                // nice sky blue. It’s initialized with constants in x and y, and then d is used for z and w. d.xy goes into c.zw. 
                // That gives c the 0.8 value in the z field.
                //  Note that c.w is used to calculate O.w (O.a) but that the alpha channel of the output pixel value is currently ignored
                float4 c= float4(.6,.7,d.x,d.y);

                // O is the output color (initialized to blue sky)
                // subtracts d.w which is the pixel’s ray march direction on the screen y axis. 
                // This has a nice effect of making a nice sky blue gradient.
                float4 O=c-d.w;

                for(float f,s,t=10.00+sin(dot(uv,uv));--t>0.;ray=.05*mul(t,d))
                    ray.xz+=_Time.y,
                    s=2.,
                    f=ray.w+1.-T-T-T-T,  // uses T macro from above
                    f<0.?O+=(O-1.-f*c.zyxw)*f*.4:O;
                return O;
            }
            ENDCG
        }
    }
}

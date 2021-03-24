Shader "Skybox/FoggyMountain"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _BaseAlt ("Noise Scale", Float ) = 600.
        _BaseSeed ("Noise Seed", Float ) = 20.
        _CloudAlt ("Cloud Fequency", Float ) = 100.0
        _CloudFactor ("Cloud Factor", Float ) = 0.00002
        _CloudFactor2 ("Cloud Factor2", Float ) = 0.00009
        _CloudSpeed ("Cloud Speed", Float ) = 50.
        _CloudOffset ("Cloud Offset", Float ) = .3
        _G1 ("Ground Factor1", Float ) = 0.00005
        _G2 ("Ground Factor2", Float ) = 10.0
        _G3 ("Mountain Height", Float ) = 40.0
        _G4 ("Ground Factor4", Float ) = 0.01
        _FogFactor ("Fog Factor", Float) = 0.0002
        _CamHeight ("Camera Height", Float) = 500.0
        _GroundColor ("Ground Color", Color) = (.3,.3,.3,1) 
        _SkyColor ("Sky Color", Color) = (0.1,.30,0.1,1) 
        _SunColor ("Sun Color", Color) = (0.1,.30,0.1,1) 
        _AmbYOffset ("Ambient YOffset", Float) = 30.
        _SunLat ("Sun Lat", Range (-1.0,1.)) = 0.5 // sliders 
        _SunHeight ("Sun Height", Range (-10,1.)) = 0.5 // sliders 
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

            // make fog work
   //         #pragma multi_compile_fog

            #include "UnityCG.cginc"

            SamplerState my_point_clamp_sampler;
            sampler2D _MainTex;
//            Texture2D _MainTex;
            float _BaseAlt;
            float _BaseSeed;
            float _CloudAlt;
            float _CloudFactor;
            float _CloudFactor2;
            float _CloudSpeed;
            float _CloudOffset;
            float _FogFactor;
            float _G1,_G2,_G3,_G4;
            float _CamHeight;
            float4 _GroundColor,_SkyColor,_SunColor;
            float _AmbYOffset;
            float _SunLat,_SunHeight;

         // Use mouse to control the camera & time.


        float time = 0.0;
             
        float3 rotate(float3 r, float v){ return float3(r.x*cos(v)+r.z*sin(v),r.y,r.z*cos(v)-r.x*sin(v));}

        float noise( in float3 x )
        {
            float  z = x.z*64.0;
            float2 offz = float2(0.317,0.123);
            float2 uv1 = x.xy + offz*floor(z); 
            float2 uv2 = uv1  + offz;
         //   return lerp(textureLod( iChannel0, uv1 ,0.0).x,textureLod( iChannel0, uv2 ,0.0).x,fract(z))-0.5;
        //    return lerp(_MainTex.SampleLevel(my_point_clamp_sampler,  uv1 ,0.0).x,_MainTex.SampleLevel(my_point_clamp_sampler,  uv2 ,0.0).x,frac(z))-0.5;
            return lerp(tex2D(_MainTex,  uv1 ).x,tex2D(_MainTex,  uv2 ).x,frac(z))-0.5;
        //    return tex2D(_MainTex,  uv1 ).x;
        }

        float noises( in float3 p){
            float a = 0.0;
            for(float i=1.0;i<3.0;i++){
                a += noise(p)/i;
                p = p*2.0 + float3(0.0,a*0.001/(i*2.),a*0.0001/(i*2.));
            }
            return a;
        }

        float base( in float3 p){
            return noise(p*_BaseSeed)*_BaseAlt;
        }

        float ground( in float3 p){
 //           return p.y + 1.;                            // horizontal plane at y = -1
//            return base(p)+noises(p.zxy*0.00005+10.0)*40.0*(0.0-p.y*0.01)+p.y;

            return base(p)+noises(p.zxy*_G1+_G2)*_G3*(0.0-p.y*_G4)+p.y;
            
        }

        float clouds( in float3 p){
            float b = base(p);
            p.y += b*0.5/abs(p.y) + _CloudAlt;

        //    return noises(float3(p.x*0.3+((time+iMouse.y)*30.0),p.y,p.z)*0.00002)-max(p.y,0.0)*0.00009;
            return noises(float3(p.x*_CloudOffset+((time)*_CloudSpeed),p.y,p.z)*_CloudFactor)-max(p.y,0.5)*_CloudFactor2;
        }

            

           struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
                float4 posWorld : TEXCOORD0;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                 o.posWorld = mul(unity_ObjectToWorld, v.vertex);
               // UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }





            fixed4 frag (v2f i) : SV_Target
            {   
                float3 viewDirection = normalize(i.posWorld.xyz- _WorldSpaceCameraPos.xyz  );
                float3 worldPos = _WorldSpaceCameraPos.xyz;

                float3 finalColor;

                time        = _Time.y*5.0;//+floor(_Time.y*0.1)*150.0;
                //time = 0.;
                float2 uv     = i.vertex.xy; 
                float3 campos  = worldPos;
                    campos.y = _CamHeight;//-base(campos);

                float3 ray = viewDirection;
                float3 pos    = campos+ray;
                float3 sun    = float3(0.0,_SunLat,_SunHeight);       
                
                // raymarch
                float test  = 0.0;
                float fog   = 0.0;
                float dist  = 0.0;

                float3  p1 = pos; 
                for(float i=1.0;i<50.0;i++){
                    test  = ground(p1); 
                    fog  += max(test*clouds(p1),fog*0.2);
                    p1   += ray*min(test,i*i*0.5);
                    dist += test;
                    if(abs(test)<10.0||dist>40000.0) break;
                }

                float l     = sin(dot(ray,sun));
                float3  light = float3(l,0.0,-l)+ray.y*0.2;

                
                // ambientLight  = p1(along ray)+AmbYOffset + sunVector*10 
//                float amb = smoothstep(-100.0,100.0,ground(p1+float3(0.0,30.0,0.0)+sun*10.0))-smoothstep(1000.0,-0.0,p1.y)*0.7;
                float amb = smoothstep(-100.0,100.0,ground(p1+float3(0.0 ,_AmbYOffset, 0.0)+sun*10.0))-smoothstep(1000.0,-0.0,p1.y)*0.7;

                // groundColor = DefaultGroundTint + sinNoise + positionNoise + AmbientLight + directLight 
//                float3  ground = float3(0.10,0.20,0.15)+sin(p1*0.001)*0.01+noise(float3(p1*0.02))*0.1+amb*0.7+light*0.01;
//                float3  ground = float3(_GroundColor.xyz)+sin(p1*0.001)*0.01+noise(float3(p1*0.01))*0.1+amb*0.7+light*0.01;
                float3  ground = float3(_GroundColor.xyz)+amb*0.7+light*0.01;
                    
                float f = smoothstep(0.0,800.0,fog);
//                float3  cloud = float3(0.70,0.72,0.70)+light*0.05+sin(fog*0.0002)*0.2+noise(p1)*0.05;
//                float3  cloud = float3(_SkyColor.xyz)+light*0.05+sin(fog*_FogFactor)*0.2+noise(p1)*0.05;
                float3  cloud = float3(_SkyColor.xyz);//+light*0.05*0.2+noise(p1)*0.05;

                float ht = smoothstep(10000.,40000.0,dist);
                float3  sky = (ray.y*0.1-0.02)+cloud;   


                float4 fragColor; //= float4(finalColor.xyz,1.0);
           //     fragColor = float4(sqrt(smoothstep(0.2,1.0,lerp(lerp(ground,sky,ht),cloud,f)-dot(uv,uv)*0.1)),1.0);
           //     fragColor = float4(lerp(ground,sky,ht),1.0);
           //     fragColor = float4(sqrt(smoothstep(0.2,1.0,lerp(lerp(ground,sky,ht),cloud,f))-noise(ray)*.01),1.0);

//           fragColor = float4(sqrt(smoothstep(0.2,1.0,lerp(ground,sky,ht))-noise(ray)*.01),1.0);
           fragColor = float4(sqrt(smoothstep(0.2,1.0,lerp(ground,sky,ht))),1.0);

//           fragColor = float4(sqrt(smoothstep(0.2,1.0,lerp(lerp(ground,sky,ht),cloud,f))-noise(ray)*.01),1.0);

           //     fragColor = float4(noises(rd),noises(rd),noises(rd),1.);
                return fragColor;
                

            }

            ENDCG
        }
    }
}

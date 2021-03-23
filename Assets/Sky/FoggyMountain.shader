Shader "Skybox/FoggyMountain"
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

            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            sampler2D _MainTex;
//            Texture2D _MainTex;

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
            SamplerState my_point_clamp_sampler;
        //    return lerp(_MainTex.SampleLevel(my_point_clamp_sampler,  uv1 ,0.0).x,_MainTex.SampleLevel(my_point_clamp_sampler,  uv2 ,0.0).x,frac(z))-0.5;
            return lerp(tex2D(_MainTex,  uv1 ).x,tex2D(_MainTex,  uv2 ).x,frac(z))-0.5;
        //    return tex2D(_MainTex,  uv1 ).x;
        }

        float noises( in float3 p){
            float a = 0.0;
            for(float i=1.0;i<6.0;i++){
                a += noise(p)/i;
                p = p*2.0 + float3(0.0,a*0.001/i,a*0.0001/i);
            }
            return a;
        }

        float base( in float3 p){
            return noise(p*0.00002)*1200.0;
        }

        float ground( in float3 p){
            //return p.y + 1.;                            // horizontal plane at y = -1
            return base(p)+noises(p.zxy*0.00005+10.0)*40.0*(0.0-p.y*0.01)+p.y;
        }

        float clouds( in float3 p){
            float b = base(p);
            p.y += b*0.5/abs(p.y) + 100.0;

        //    return noises(float3(p.x*0.3+((time+iMouse.y)*30.0),p.y,p.z)*0.00002)-max(p.y,0.0)*0.00009;
            return noises(float3(p.x*0.3+((time)*30.0),p.y,p.z)*0.00002)-max(p.y,0.0)*0.00009;
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
                //o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.posWorld = mul(unity_ObjectToWorld, v.vertex);
               // UNITY_TRANSFER_FOG(o,o.vertex);
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


            fixed4 frag (v2f i) : SV_Target
            {   
                float3 viewDirection = normalize(i.posWorld.xyz- _WorldSpaceCameraPos.xyz  );
                float3 worldPos = i.posWorld.xyz;//_WorldSpaceCameraPos.xyz;

                float3 finalColor = float4(0.5,0.4,0.6,0.);

                time        = _Time.y*5.0+floor(_Time.y*0.1)*150.0;
                //time = 0.;
                float2 uv     = i.posWorld.xy; //fragCoord.xy/(iResolution.xx*0.5)-float2(1.0,iResolution.y/iResolution.x);
                float3 campos   = float3(30.0,500.0,time*8.0);
                     //campos = worldPos;
                     campos.y = 500.0-base(campos);

            //    float3 ray   = rotate(normalize(float3(uv.x,uv.y-sin(time*0.05)*0.2-0.1,1.0).xyz),time*0.01+iMouse.x*0.009);
                float3 ray   = rotate(normalize(float3(uv.x,uv.y-sin(time*0.05)*0.2-0.1,1.0).xyz),time*0.01+worldPos.x*0.009);
                ray = viewDirection;
                float3 pos    = campos+ray;
                float3 sun    = float3(0.0,600.,-400.);       
                
                // raymarch
                float test  = 0.0;
                float fog   = 0.0;
                float dist  = 0.0;

                float3  p1 = pos; 
                for(float i=1.0;i<50.0;i++){
                    test  = ground(p1); 
                    fog  += max(test*clouds(p1),fog*0.02);
                    p1   += ray*min(test,i*i*0.5);
                    dist += test;
                    if(abs(test)<10.0||dist>40000.0) break;
                }

                float l     = sin(dot(ray,sun));
                float3  light = float3(l,0.0,-l)+ray.y*0.2;
                
                float amb = smoothstep(-100.0,100.0,ground(p1+float3(0.0,30.0,0.0)+sun*10.0))-smoothstep(1000.0,-0.0,p1.y)*0.7;
                float3  ground = float3(0.30,0.30,0.25)+sin(p1*0.001)*0.01+noise(float3(p1*0.02))*0.1+amb*0.7+light*0.01;
                    
                float f = smoothstep(0.0,800.0,fog);
                float3  cloud = float3(0.70,0.72,0.70)+light*0.05+sin(fog*0.0002)*0.2+noise(p1)*0.05;

                float ht = smoothstep(10000.,40000.0,dist);
                float3  sky = cloud+ray.y*0.1-0.02;   


                float4 fragColor; //= float4(finalColor.xyz,1.0);
           //     fragColor = float4(sqrt(smoothstep(0.2,1.0,lerp(lerp(ground,sky,ht),cloud,f)-dot(uv,uv)*0.1)),1.0);
           //     fragColor = float4(lerp(ground,sky,ht),1.0);
           fragColor = float4(sqrt(smoothstep(0.2,1.0,lerp(lerp(ground,sky,ht),cloud,f))-noise(ray)*.1),1.0);
           //     fragColor = float4(noises(rd),noises(rd),noises(rd),1.);
                return fragColor;
                

            }

            ENDCG
        }
    }
}

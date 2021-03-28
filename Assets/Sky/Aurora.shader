Shader "Skybox/Aurora"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
         [MaterialToggle] _StereoEnabled ("Stereo Enabled", Float ) = 0
        _SkyColor1 ("SkyColor1", Color) = (1.0,0.49,0.1,1) 
        _SkyColor2 ("SkyColor2", Color) = (0.75,0.9,1.,1) 
        _WaterColor1 ("WaterColor1", Color) = (0.2,0.25,0.5,1) 
        _WaterColor2 ("WaterColor2", Color) = (0.1,0.05,0.2,1) 

   
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
            uniform float4 _MainTex_ST;
            uniform fixed _StereoEnabled;
            float4 _SkyColor1,_SkyColor2,_WaterColor1,_WaterColor2;


            float2x2 rotate(in float a){ float c = cos(a), s = sin(a);return float2x2(c,s,-s,c);}
            float2x2 m2 = float2x2(0.95534, 0.29552, -0.29552, 0.95534);
            float tri(in float x){return clamp(abs(frac(x)-.5),0.01,0.49);}
            float2 tri2(in float2 p){return float2(tri(p.x)+tri(p.y),tri(p.y+tri(p.x)));}

            float triNoise2d(in float2 p, float spd)
            {
                float z=1.8;
                float z2=2.5;
                float rz = 0.;
                p = mul(p,rotate(p.x*0.06));
                float2 bp = p;
                for (float i=0.; i<5.; i++ )
                {
                    float2 dg = tri2(bp*1.85)*.75;
                    dg = mul(dg,rotate(_Time.y*spd));
                    p -= dg/z2;

                    bp *= 1.3;
                    z2 *= .45;
                    z *= .42;
                    p *= 1.21 + (rz-1.0)*.02;
                    
                    rz += tri(p.x+tri(p.y))*z;
                    p = mul(p, -m2);
                }
                return clamp(1./pow(rz*29., 1.3),0.,.55);
            }

            float hash21(in float2 n){ return frac(sin(dot(n, float2(12.9898, 4.1414))) * 43758.5453); }
            float4 aurora(float3 ro, float3 rd, float2 uv)
            {
                float4 col = float4(0,0,0,0);
                float4 avgCol = float4(0,0,0,0);
                
                for(float i=0.;i<25.;i++)
                {
                    float of = 0.006*hash21(uv)*smoothstep(0.,15., i*2);
                    float pt = ((.8+pow(i*2,1.4)*.002)+ro.y*.001)/(rd.y*2.+0.4);
                    pt -= of;
                    float3 bpos = ro + mul(pt,rd);
                    float2 p = bpos.zx;
                    float rzt = triNoise2d(p, 0.06);
                    float4 col2 = float4(0,0,0, rzt);
                    col2.rgb = (sin(1.-float3(2.15,-.5, 1.2)+i*2*0.043)*0.5+0.5)*rzt;
                    avgCol =  lerp(avgCol, col2, .5);
                    col += avgCol*exp2(-i*2*0.065 - 2.5)*smoothstep(0.,5., i);
                    
                }
                
                col *= (clamp(rd.y*15.+.4,0.,1.));

                return col*1.8;
            }

            //-------------------Background and Stars--------------------

            float3 nmzHash33(float3 q)
            {
                uint3 p = uint3(int3(q));
                p = mul(p,uint3(374761393U, 1103515245U, 668265263U)) + p.zxy + p.yzx;
                p = mul(p.yzx,(p.zxy^(p >> 3U)));
                uint3 m = (0xffffffffU,0xffffffffU,0xffffffffU);
                uint r = p^(p >> 16U);
                uint3 r3 = uint3(r,r,r);

                return mul(r3,(1.0/m));
            }


            float3 stars(in float3 p)
            {
                float3 c = float3(0.,0.,0.);
                float res = 1000.; //iResolution.x*1.;
                
                for (float i=0.;i<2.;i++)
                {
                    float3 q = frac(p*(.15*res))-0.5;
                    float3 id = floor(p*(.15*res));
                    float2 rn = nmzHash33(id).xy;
                    float c2 = 1.-smoothstep(0.,.6,length(q));
                    c2 *= step(rn.x,.0005+i*i*0.001);
                    c += c2*(lerp(float3(1.0,0.49,0.1),float3(0.75,0.9,1.),rn.y)*0.1+0.9);
                    p *= 1.3;
                }
                return c*c*.8;
            }

            float3 bg(in float3 rd)
            {
            return 0;
             //   float sd = dot(normalize(float3(-0.5, -0.6, 0.9)), rd)*0.5+0.5;
             //   sd = pow(sd, 5.);
             //   float3 col = lerp(float3(0.05,0.1,0.2), float3(0.1,0.05,0.2), sd);
             //   return col*.63;
            }




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
            


            

            float4 frag(VertexOutput v) : SV_TARGET //COLOR 
            {

                float3 viewDirection = normalize(v.posWorld.xyz- _WorldSpaceCameraPos.xyz  );

                float3 finalColor = float4(0.5,0.5,0.5,1.);

                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy



                float2 uv = v.posWorld.xz*.0001;
                
                
                float3 col = float3(0.,0.,0.);
                float3 brd = rd;
                float fade = smoothstep(0.,0.01,abs(brd.y))*0.1+0.9;
                
                col = bg(rd)*fade;
                
                if (rd.y > 0.) {
                    float4 aur = smoothstep(0.,1.5,aurora(ro,rd,uv))*fade;
                    col += stars(rd);
                    col = col*(1.-aur.a) + aur.rgb;
                } 
                    else //Reflections
                {
                    rd.y = abs(rd.y);
                    col = bg(rd)*fade*0.6;
                    float4 aur = smoothstep(0.0,2.5,aurora(ro,rd,uv));
                    col += stars(rd)*0.1;
                    col = col*(1.-aur.a) + aur.rgb;
                    float3 pos = ro + ((0.5-ro.y)/rd.y)*rd;
                    float nz2 = triNoise2d(pos.xz*float2(.5,.7), 0.);
                    col += lerp(float3(_WaterColor1.xyz)*0.08,float3(_WaterColor2.xyz)*0.7, nz2*0.4);

                }
                finalColor = float4(col.x,col.y,col.z, 1.);



                return fixed4(finalColor,1);
            }
  
            ENDCG
        }
    }
}

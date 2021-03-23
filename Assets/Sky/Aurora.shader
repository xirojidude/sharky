Shader "Skybox/Aurora"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
         [MaterialToggle] _StereoEnabled ("Stereo Enabled", Float ) = 0
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


            float2x2 mm2(in float a){ float c = cos(a), s = sin(a);return float2x2(c,s,-s,c);}
            float2x2 m2 = float2x2(0.95534, 0.29552, -0.29552, 0.95534);
            float tri(in float x){return clamp(abs(frac(x)-.5),0.01,0.49);}
            float2 tri2(in float2 p){return float2(tri(p.x)+tri(p.y),tri(p.y+tri(p.x)));}

            float triNoise2d(in float2 p, float spd)
            {
                float z=1.8;
                float z2=2.5;
                float rz = 0.;
                p = mul(p,mm2(p.x*0.06));
                float2 bp = p;
                for (float i=0.; i<5.; i++ )
                {
                    float2 dg = tri2(bp*1.85)*.75;
                    dg = mul(dg,mm2(_Time.y*spd));
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
    
    for(float i=0.;i<50.;i++)
    {
        float of = 0.006*hash21(uv)*smoothstep(0.,15., i);
        float pt = ((.8+pow(i,1.4)*.002)-ro.y)/(rd.y*2.+0.4);
        pt -= of;
        float3 bpos = ro + mul(pt,rd);
        float2 p = bpos.zx;
        float rzt = triNoise2d(p, 0.06);
        float4 col2 = float4(0,0,0, rzt);
        col2.rgb = (sin(1.-float3(2.15,-.5, 1.2)+i*0.043)*0.5+0.5)*rzt;
        avgCol =  lerp(avgCol, col2, .5);
        col += avgCol*exp2(-i*0.065 - 2.5)*smoothstep(0.,5., i);
        
    }
    
    col *= (clamp(rd.y*15.+.4,0.,1.));
    
    
    //return clamp(pow(col,float4(1.3))*1.5,0.,1.);
    //return clamp(pow(col,float4(1.7))*2.,0.,1.);
    //return clamp(pow(col,float4(1.5))*2.5,0.,1.);
    //return clamp(pow(col,float4(1.8))*1.5,0.,1.);
    
    //return smoothstep(0.,1.1,pow(col,float4(1.))*1.5);
    return col*1.8;
    //return pow(col,float4(1.))*2.
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
    //    return vec3(p^(p >> 16U))*(1.0/vec3(0xffffffffU));

    return mul(r3,(1.0/m));
}


float3 stars(in float3 p)
{
    float3 c = float3(0.,0.,0.);
    float res = 1000.; //iResolution.x*1.;
    
    for (float i=0.;i<4.;i++)
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
    float sd = dot(normalize(float3(-0.5, -0.6, 0.9)), rd)*0.5+0.5;
    sd = pow(sd, 5.);
    float3 col = lerp(float3(0.05,0.1,0.2), float3(0.1,0.05,0.2), sd);
    return col*.63;
}



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
                float latitude = 1.-(acos(normalizedCoords.y)/UNITY_PI);
                float longitude = 1.-clamp(atan2(normalizedCoords.z, normalizedCoords.x),-3.145926535,3.145926535)/UNITY_PI;

                float2 sphereCoords = float2(longitude, latitude); // * float2(1.0/UNITY_PI, 1.0/UNITY_PI);
                //sphereCoords = float2(1.0,1.) - sphereCoords;
                //return sphereCoords;
                return (sphereCoords + float4(0, 1-unity_StereoEyeIndex,1,1.0).xy) * float4(0, 1-unity_StereoEyeIndex,1,1.0).zw;
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
            

            #define TRANSFORM_TEX2(tex,name) (float2(0.5, 1.0) * tex.xy * name##_ST.xy + name##_ST.zw)


            //
            // Distance field function for the scene. It combines
            // the seperate distance field functions of three spheres
            // and a plane using the min-operator.
            //

            float map(float3 p) {
                float d = distance(p, float3(-10, 0, -50)) - 5.;     // sphere at (-1,0,5) with radius 1
                d = min(d, distance(p, float3(20, 0, -30)) - 10.);    // second sphere
                d = min(d, distance(p, float3(-20, 0, -20)) - 10.);   // and another
              //  d = min(d, p.y + 1.);                            // horizontal plane at y = -1
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

            

            float4 frag(VertexOutput v) : SV_TARGET //COLOR 
            {





                float3 viewDirection = normalize(v.posWorld.xyz- _WorldSpaceCameraPos.xyz  );

                float3 finalColor = float4(1.,0.,0.,0.);
                float2 _StereoEnabled_var;
                float4 _MainTex_var;
                _StereoEnabled_var = MonoPanoProjection( viewDirection );
                _MainTex_var = tex2D(_MainTex,TRANSFORM_TEX2(_StereoEnabled_var, _MainTex));
                finalColor = _MainTex_var.rgb;

                float3 ro = _WorldSpaceCameraPos.xyz; //v.posWorld.xyz;                           // ray origin

                float3 rd = viewDirection;             // ray direction for fragCoord.xy



    float2 uv = v.posWorld.xy;
    float2 q = float2(1.0,1.0); //fragCoord.xy / iResolution.xy;
    float2 p = q - 0.5;
    p.x*=1.0; //iResolution.x/iResolution.y;
    
    float3 rc = float3(0.,0.,-6.7);
    float3 re = normalize(float3(p,1.3));
    float2 mo = _WorldSpaceCameraPos.xy;    //iMouse.xy / iResolution.xy-.5;
  //  if (mo==float2(-.5,-.5)) mo=float2(-0.1,0.1);
    mo.x *= 1.0;   //iResolution.x/iResolution.y;
    rd.yz = mul(rd.yz,mm2(mo.y));
    rd.xz = mul(rd.xz, mm2(mo.x + sin(_Time.y*0.05)*0.2));
    
    float3 col = float3(0.,0.,0.);
    float3 brd = re;
    float fade = smoothstep(0.,0.01,abs(brd.y))*0.1+0.9;
    
    col = bg(re)*fade;
    
    if (re.y > 0.){
        float4 aur = smoothstep(0.,1.5,aurora(rc,re,uv))*fade;
        col += stars(re);
        col = col*(1.-aur.a) + aur.rgb;
    }
finalColor = float4(col.x,col.y,col.z, 1.);




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


                if (viewDirection.z > -.001 && viewDirection.z < .001 && viewDirection.x <0) {
                    finalColor=float4(1.,0.,0.,1.);
                }
                if (viewDirection.y  <0) {
                    //finalColor=float4(ro.x,ro.x,ro.x,1.);
                }
                return fixed4(finalColor,1);
            }
  
            ENDCG
        }
    }
}
